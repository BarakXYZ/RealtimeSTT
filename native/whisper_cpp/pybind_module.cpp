#include <atomic>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "whisper.h"

namespace py = pybind11;

namespace {

struct WhisperSegment {
    std::string text;
    std::int64_t t0_ms;
    std::int64_t t1_ms;
};

struct WhisperTranscription {
    std::string text;
    std::vector<WhisperSegment> segments;
    std::string language;
    double language_probability;
};

class WhisperCppModel;

class WhisperCppState {
  public:
    explicit WhisperCppState(std::shared_ptr<WhisperCppModel> model);
    ~WhisperCppState();

    WhisperTranscription transcribe(
        py::array_t<float, py::array::c_style | py::array::forcecast> audio,
        const std::string & language,
        bool detect_language,
        const std::string & initial_prompt,
        int beam_size,
        bool no_context,
        bool single_segment,
        bool no_timestamps,
        bool token_timestamps,
        int n_threads);

    void request_abort();

  private:
    std::shared_ptr<WhisperCppModel> model_;
    whisper_state * state_;
    std::mutex decode_mutex_;
    std::atomic<bool> abort_requested_{false};
};

class WhisperCppModel : public std::enable_shared_from_this<WhisperCppModel> {
  public:
    WhisperCppModel(
        const std::string & model_path,
        bool use_gpu,
        int gpu_device,
        bool flash_attn,
        const std::string & openvino_encoder_path,
        const std::string & openvino_device,
        const std::string & openvino_cache_dir
    ) : model_path_(model_path),
        openvino_encoder_path_(openvino_encoder_path),
        openvino_device_(openvino_device),
        openvino_cache_dir_(openvino_cache_dir) {
        whisper_context_params context_params = whisper_context_default_params();
        context_params.use_gpu = use_gpu;
        context_params.flash_attn = flash_attn;
        context_params.gpu_device = gpu_device;

        context_ = whisper_init_from_file_with_params_no_state(model_path.c_str(), context_params);
        if (context_ == nullptr) {
            throw std::runtime_error("Failed to initialize whisper.cpp model from: " + model_path);
        }
    }

    ~WhisperCppModel() {
        if (context_ != nullptr) {
            whisper_free(context_);
        }
    }

    std::shared_ptr<WhisperCppState> create_state() {
        return std::make_shared<WhisperCppState>(shared_from_this());
    }

    whisper_context * context() const {
        return context_;
    }

    const std::string & model_path() const {
        return model_path_;
    }

    const std::string & openvino_encoder_path() const {
        return openvino_encoder_path_;
    }

    const std::string & openvino_device() const {
        return openvino_device_;
    }

    const std::string & openvino_cache_dir() const {
        return openvino_cache_dir_;
    }

  private:
    std::string model_path_;
    std::string openvino_encoder_path_;
    std::string openvino_device_;
    std::string openvino_cache_dir_;
    whisper_context * context_ = nullptr;
};

bool whisper_abort_callback(void * user_data) {
    auto * flag = static_cast<std::atomic<bool> *>(user_data);
    return flag->load();
}

WhisperCppState::WhisperCppState(std::shared_ptr<WhisperCppModel> model) : model_(std::move(model)) {
    state_ = whisper_init_state(model_->context());
    if (state_ == nullptr) {
        throw std::runtime_error("Failed to initialize whisper.cpp state");
    }

    if (!model_->openvino_encoder_path().empty()) {
        const char * cache_dir = model_->openvino_cache_dir().empty() ? nullptr : model_->openvino_cache_dir().c_str();
        int rc = whisper_ctx_init_openvino_encoder_with_state(
            model_->context(),
            state_,
            model_->openvino_encoder_path().c_str(),
            model_->openvino_device().empty() ? "CPU" : model_->openvino_device().c_str(),
            cache_dir
        );
        if (rc != 0) {
            throw std::runtime_error("Failed to initialize OpenVINO encoder for whisper.cpp");
        }
    }
}

WhisperCppState::~WhisperCppState() {
    if (state_ != nullptr) {
        whisper_free_state(state_);
    }
}

WhisperTranscription WhisperCppState::transcribe(
    py::array_t<float, py::array::c_style | py::array::forcecast> audio,
    const std::string & language,
    bool detect_language,
    const std::string & initial_prompt,
    int beam_size,
    bool no_context,
    bool single_segment,
    bool no_timestamps,
    bool token_timestamps,
    int n_threads
) {
    std::lock_guard<std::mutex> lock(decode_mutex_);
    abort_requested_.store(false);

    py::buffer_info info = audio.request();
    if (info.ndim != 1) {
        throw std::runtime_error("Audio array must be one-dimensional float32 PCM");
    }

    auto * samples = static_cast<float *>(info.ptr);
    auto sample_count = static_cast<int>(info.shape[0]);
    if (sample_count <= 0) {
        throw std::runtime_error("Audio array must not be empty");
    }

    whisper_full_params params = whisper_full_default_params(
        beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY
    );
    params.n_threads = n_threads > 0 ? n_threads : static_cast<int>(std::thread::hardware_concurrency());
    params.print_progress = false;
    params.print_special = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.no_context = no_context;
    params.single_segment = single_segment;
    params.no_timestamps = no_timestamps;
    params.token_timestamps = token_timestamps;
    params.language = language.empty() || language == "auto" ? nullptr : language.c_str();
    params.detect_language = detect_language || language.empty() || language == "auto";
    params.initial_prompt = initial_prompt.empty() ? nullptr : initial_prompt.c_str();
    params.abort_callback = whisper_abort_callback;
    params.abort_callback_user_data = &abort_requested_;
    params.beam_search.beam_size = beam_size;
    params.greedy.best_of = 1;

    {
        py::gil_scoped_release release;
        const int rc = whisper_full_with_state(model_->context(), state_, params, samples, sample_count);
        if (rc != 0) {
            if (abort_requested_.load()) {
                return WhisperTranscription{};
            }
            throw std::runtime_error("whisper.cpp transcription failed");
        }
    }

    WhisperTranscription output;
    int language_id = whisper_full_lang_id_from_state(state_);
    if (language_id >= 0) {
        output.language = whisper_lang_str(language_id);
    }
    output.language_probability = 0.0;

    int segment_count = whisper_full_n_segments_from_state(state_);
    output.segments.reserve(segment_count);
    for (int i = 0; i < segment_count; ++i) {
        const char * segment_text = whisper_full_get_segment_text_from_state(state_, i);
        WhisperSegment segment{
            segment_text == nullptr ? "" : std::string(segment_text),
            whisper_full_get_segment_t0_from_state(state_, i) * 10,
            whisper_full_get_segment_t1_from_state(state_, i) * 10,
        };
        if (!segment.text.empty()) {
            if (!output.text.empty()) {
                output.text += " ";
            }
            output.text += segment.text;
        }
        output.segments.push_back(std::move(segment));
    }

    return output;
}

void WhisperCppState::request_abort() {
    abort_requested_.store(true);
}

}  // namespace

PYBIND11_MODULE(_whisper_cpp_native, module) {
    module.doc() = "RealtimeSTT whisper.cpp native integration";

    py::class_<WhisperSegment>(module, "WhisperSegment")
        .def_readonly("text", &WhisperSegment::text)
        .def_readonly("t0_ms", &WhisperSegment::t0_ms)
        .def_readonly("t1_ms", &WhisperSegment::t1_ms);

    py::class_<WhisperTranscription>(module, "WhisperTranscription")
        .def_readonly("text", &WhisperTranscription::text)
        .def_readonly("segments", &WhisperTranscription::segments)
        .def_readonly("language", &WhisperTranscription::language)
        .def_readonly("language_probability", &WhisperTranscription::language_probability);

    py::class_<WhisperCppModel, std::shared_ptr<WhisperCppModel>>(module, "WhisperCppModel")
        .def(
            py::init<
                const std::string &,
                bool,
                int,
                bool,
                const std::string &,
                const std::string &,
                const std::string &
            >(),
            py::arg("model_path"),
            py::arg("use_gpu") = true,
            py::arg("gpu_device") = 0,
            py::arg("flash_attn") = false,
            py::arg("openvino_encoder_path") = "",
            py::arg("openvino_device") = "CPU",
            py::arg("openvino_cache_dir") = ""
        )
        .def("create_state", &WhisperCppModel::create_state)
        .def_property_readonly("model_path", &WhisperCppModel::model_path);

    py::class_<WhisperCppState, std::shared_ptr<WhisperCppState>>(module, "WhisperCppState")
        .def(
            "transcribe",
            &WhisperCppState::transcribe,
            py::arg("audio"),
            py::arg("language") = "",
            py::arg("detect_language") = false,
            py::arg("initial_prompt") = "",
            py::arg("beam_size") = 5,
            py::arg("no_context") = true,
            py::arg("single_segment") = false,
            py::arg("no_timestamps") = true,
            py::arg("token_timestamps") = false,
            py::arg("n_threads") = 0
        )
        .def("request_abort", &WhisperCppState::request_abort);
}
