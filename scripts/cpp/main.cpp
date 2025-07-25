#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h> // ONNX Runtime C++ API

// ここではエラー処理は省略しています

int main() {
    // 1. ONNX Runtime 環境とセッションの作成
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLODetection");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1); // スレッド数を設定（ラズパイのコア数に応じて調整）

    // ONNXモデルのパス
    const char* model_path = "../models/yolo11n.pt";
    Ort::Session session(env, model_path, session_options);

    // モデルの入出力情報を取得
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();

    // 入力名とシェイプの取得
    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
    const char* input_name = input_name_ptr.get();
    Ort::TypeInfo input_type_info = session.GetInputTypeInfo(0);
    auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_dims = input_tensor_info.GetShape();

    // 出力名とシェイプの取得
    Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* output_name = output_name_ptr.get();
    // ... 出力シェイプの取得も同様に行う

    std::vector<const char*> input_names = {input_name};
    std::vector<const char*> output_names = {output_name};

    // 2. Webカメラの起動
    cv::VideoCapture cap(0); // 0はデフォルトのWebカメラ
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // 3. 入力データの前処理（例: YOLOの入力形式に合わせる）
        // 解像度調整、BGRからRGB、チャネル順序変更、正規化など
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, cv::Size(input_dims[3], input_dims[2])); // 例: 640x640
        
        cv::Mat float_frame;
        resized_frame.convertTo(float_frame, CV_32FC3, 1.0 / 255.0); // 0-1に正規化
        
        // BGRからRGBへ変換 (YOLOの多くはRGBを期待)
        cv::cvtColor(float_frame, float_frame, cv::COLOR_BGR2RGB);

        // HWCからCHWへ変換 (NHWC to NCHW)
        // ONNX RuntimeはNCHW形式を期待する場合が多い
        std::vector<float> input_tensor_values(
            float_frame.channels * float_frame.rows * float_frame.cols
        );
        for (int c = 0; c < float_frame.channels; ++c) {
            for (int h = 0; h < float_frame.rows; ++h) {
                for (int w = 0; w < float_frame.cols; ++w) {
                    // ここでチャネル順序を適切に並べ替える
                    // 例: RGBの場合 (R,G,B)を(R,R,R,...,G,G,G,...,B,B,B...)のように並べる
                    input_tensor_values[c * float_frame.rows * float_frame.cols + h * float_frame.cols + w] = 
                        float_frame.at<cv::Vec3f>(h, w)[c]; // OpenCVはBGR順なので注意
                }
            }
        }
        
        // 入力テンソルの作成
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_tensor_info,
            input_tensor_values.data(), 
            input_tensor_values.size(), 
            input_dims.data(), 
            input_dims.size()
        );

        // 4. 推論の実行
        std::vector<Ort::Value> output_tensors = session.Run(
            Ort::RunOptions{nullptr}, 
            input_names.data(), 
            &input_tensor, 
            1, 
            output_names.data(), 
            1
        );

        // 5. 出力データの後処理
        // output_tensors[0] からバウンディングボックス、クラス、信頼度を抽出し、NMSなどを行う
        // ... (YOLOの出力フォーマットに応じた複雑な処理が必要) ...

        // 6. 結果の描画
        // cv::rectangle, cv::putText などを使ってフレームに描画
        // ...

        cv::imshow("YOLO Detection", frame); // 描画されたフレームを表示

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}