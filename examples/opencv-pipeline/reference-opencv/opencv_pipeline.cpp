// SPDX-License-Identifier: MIT

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Config {
    std::filesystem::path input;
    std::filesystem::path out = ".tmp/opencv-pipeline/opencv";
    double sigma = 1.5;
    int min_size = 20;
    std::string mode = "segment";
};

struct StageTimings {
    double transform_seconds = 0.0;
    double smooth_seconds = 0.0;
    double threshold_seconds = 0.0;
    double morphology_seconds = 0.0;
    double filter_seconds = 0.0;
    double label_seconds = 0.0;
    double measure_seconds = 0.0;
};

std::string value(int argc, char** argv, int& i, const char* name) {
    ++i;
    if (i >= argc) {
        throw std::invalid_argument(std::string(name) + " needs a value");
    }
    return argv[i];
}

Config parse(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input") {
            config.input = value(argc, argv, i, "--input");
        } else if (arg == "--out") {
            config.out = value(argc, argv, i, "--out");
        } else if (arg == "--sigma") {
            config.sigma = std::stod(value(argc, argv, i, "--sigma"));
        } else if (arg == "--min-size") {
            config.min_size = std::stoi(value(argc, argv, i, "--min-size"));
        } else if (arg == "--mode") {
            config.mode = value(argc, argv, i, "--mode");
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "opencv_pipeline --input IMAGE --out DIR [--mode segment|transform]\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument " + arg);
        }
    }
    if (config.input.empty()) {
        throw std::invalid_argument("missing required --input");
    }
    if (!(config.sigma >= 0.0)) {
        throw std::invalid_argument("--sigma must be non-negative");
    }
    if (config.min_size < 1) {
        throw std::invalid_argument("--min-size must be at least 1");
    }
    if (config.mode != "segment" && config.mode != "transform") {
        throw std::invalid_argument("--mode must be segment or transform");
    }
    return config;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double>(elapsed).count();
}

cv::Mat transform_if_requested(const cv::Mat& input, const std::string& mode) {
    if (mode == "segment") {
        return input.clone();
    }
    cv::Mat matrix = (cv::Mat_<double>(2, 3) << 1.0, 0.0, 7.0, 0.0, 1.0, 5.0);
    cv::Mat out;
    cv::warpAffine(input, out, matrix, input.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
    return out;
}

void write_objects(const cv::Mat& stats, const cv::Mat& centroids, const std::filesystem::path& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("create CSV failed: " + path.string());
    }
    out << "label,count,centroid_y,centroid_x\n";
    for (int label = 1; label < stats.rows; ++label) {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area == 0) {
            continue;
        }
        out << label << ',' << area << ',' << centroids.at<double>(label, 1) << ','
            << centroids.at<double>(label, 0) << '\n';
    }
}

void write_summary(const Config& config, int objects, long long total_area, double threshold,
                   double load_seconds, double pipeline_seconds, const StageTimings& timings,
                   const std::filesystem::path& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("create summary failed: " + path.string());
    }
    out << "{\n";
    out << "  \"input\": \"" << config.input.string() << "\",\n";
    out << "  \"mode\": \"" << config.mode << "\",\n";
    out << "  \"objects\": " << objects << ",\n";
    out << "  \"total_foreground_area\": " << total_area << ",\n";
    out << "  \"threshold\": " << threshold << ",\n";
    out << "  \"sigma\": " << config.sigma << ",\n";
    out << "  \"min_size\": " << config.min_size << ",\n";
    out << "  \"load_seconds\": " << load_seconds << ",\n";
    out << "  \"pipeline_seconds\": " << pipeline_seconds << ",\n";
    out << "  \"stage_seconds\": {\n";
    out << "    \"transform\": " << timings.transform_seconds << ",\n";
    out << "    \"smooth\": " << timings.smooth_seconds << ",\n";
    out << "    \"threshold\": " << timings.threshold_seconds << ",\n";
    out << "    \"morphology\": " << timings.morphology_seconds << ",\n";
    out << "    \"filter\": " << timings.filter_seconds << ",\n";
    out << "    \"label\": " << timings.label_seconds << ",\n";
    out << "    \"measure\": " << timings.measure_seconds << "\n";
    out << "  }\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Config config = parse(argc, argv);
        std::filesystem::create_directories(config.out);

        auto started = std::chrono::steady_clock::now();
        cv::Mat input = cv::imread(config.input.string(), cv::IMREAD_GRAYSCALE);
        if (input.empty()) {
            throw std::runtime_error("could not decode " + config.input.string());
        }
        double load_seconds = seconds_since(started);

        auto pipeline_started = std::chrono::steady_clock::now();
        StageTimings timings;
        auto stage_started = std::chrono::steady_clock::now();
        cv::Mat prepared = transform_if_requested(input, config.mode);
        timings.transform_seconds = seconds_since(stage_started);
        stage_started = std::chrono::steady_clock::now();
        cv::Mat smoothed;
        if (config.sigma == 0.0) {
            smoothed = prepared;
        } else {
            cv::GaussianBlur(prepared, smoothed, cv::Size(), config.sigma, config.sigma, cv::BORDER_REFLECT);
        }
        timings.smooth_seconds = seconds_since(stage_started);

        stage_started = std::chrono::steady_clock::now();
        cv::Mat raw_mask;
        double threshold = cv::threshold(smoothed, raw_mask, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
        timings.threshold_seconds = seconds_since(stage_started);
        stage_started = std::chrono::steady_clock::now();
        cv::Mat morphed;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(raw_mask, morphed, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::morphologyEx(morphed, morphed, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1, cv::BORDER_CONSTANT, cv::Scalar(0));
        timings.morphology_seconds = seconds_since(stage_started);

        stage_started = std::chrono::steady_clock::now();
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        int components = cv::connectedComponentsWithStats(morphed, labels, stats, centroids, 4);
        cv::Mat kept = cv::Mat::zeros(morphed.size(), CV_8U);
        for (int label = 1; label < components; ++label) {
            int area = stats.at<int>(label, cv::CC_STAT_AREA);
            if (area >= config.min_size) {
                kept.setTo(255, labels == label);
            }
        }
        timings.filter_seconds = seconds_since(stage_started);

        stage_started = std::chrono::steady_clock::now();
        components = cv::connectedComponentsWithStats(kept, labels, stats, centroids, 4);
        timings.label_seconds = seconds_since(stage_started);
        stage_started = std::chrono::steady_clock::now();
        int objects = components - 1;
        long long total_area = 0;
        for (int label = 1; label < components; ++label) {
            total_area += stats.at<int>(label, cv::CC_STAT_AREA);
        }
        timings.measure_seconds = seconds_since(stage_started);
        double pipeline_seconds = seconds_since(pipeline_started);

        write_objects(stats, centroids, config.out / "objects.csv");
        write_summary(config, objects, total_area, threshold, load_seconds, pipeline_seconds, timings, config.out / "summary.json");
        std::cout << "objects=" << objects << " threshold=" << threshold << " output=" << config.out << "\n";
        return 0;
    } catch (const std::exception& err) {
        std::cerr << "opencv-pipeline: " << err.what() << "\n";
        return 1;
    }
}
