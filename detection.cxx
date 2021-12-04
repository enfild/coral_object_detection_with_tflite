#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/opencv.hpp>

#include "edgetpu.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(const tflite::FlatBufferModel &model,
                                                               edgetpu::EdgeTpuContext *edgetpu_context)
{
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::unique_ptr<tflite::Interpreter> interpreter;
  if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk)
  {
    std::cerr << "Failed to build interpreter." << std::endl;
  }
  // Bind given context with interpreter.
  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  interpreter->SetNumThreads(1);
  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    std::cerr << "Failed to allocate tensors." << std::endl;
  }
  return interpreter;
}

int main(int argc, char* argv[]) {
  std::string model_path = "detect_edgetpu.tflite";
  auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

  std::unique_ptr<edgetpu::EdgeTpuContext> tpu_context =
    edgetpu::EdgeTpuManager::GetSingleton()->NewEdgeTpuContext();
  std::cout << "Checking readiness of Coral device" << std::endl;
  if(!tpu_context->IsReady())
  {
    throw -1;
  }
  std::cout << "EDGE TPU path: " << tpu_context->GetDeviceEnumRecord().path << std::endl;
  
  std::unique_ptr<tflite::Interpreter> interpreter =
      BuildEdgeTpuInterpreter(*model, tpu_context.get());

  interpreter->SetExternalContext(kTfLiteEdgeTpuContext, tpu_context.get());
  interpreter->SetNumThreads(1);

  interpreter->AllocateTensors();

  TfLiteTensor* input_tensor = interpreter->tensor(interpreter->inputs()[0]);
  TfLiteTensor* output_locations = interpreter->tensor(interpreter->outputs()[0]);
  TfLiteTensor* output_classes = interpreter->tensor(interpreter->outputs()[1]);
  TfLiteTensor* output_scores = interpreter->tensor(interpreter->outputs()[2]);
  TfLiteTensor* num_detections_ = interpreter->tensor(interpreter->outputs()[3]);

  auto cap = cv::VideoCapture(0);

  const int height = input_tensor->dims->data[1];
  const int width = input_tensor->dims->data[2];
  const int channels = input_tensor->dims->data[3];
  const int row_elems = width * channels;

  cv::Mat frame, resized;
  int k = -1;
  while (k == -1) {
    bool ret = cap.read(frame);
    if (ret) {
      cv::resize(frame, resized, cv::Size(width, height));
      uint8_t* dst = input_tensor->data.uint8;
      for (int row = 0; row < height; row++) {
        memcpy(dst, resized.ptr(row), row_elems);
        dst += row_elems;
      }
      interpreter->Invoke();
      const float* detection_locations = output_locations->data.f;
      const float* detection_classes = output_classes->data.f;
      const float* detection_scores = output_scores->data.f;
      const int num_detections = *(num_detections_->data.f);
      
      for (int i = 0; i < num_detections; i++) {
        const float score = detection_scores[i];
        const std::string label = std::to_string(uint8_t(detection_classes[i]));

        const int ymin = detection_locations[4 * i + 0] * frame.rows;
        const int xmin = detection_locations[4 * i + 1] * frame.cols;
        const int ymax = detection_locations[4 * i + 2] * frame.rows;
        const int xmax = detection_locations[4 * i + 3] * frame.cols;
        if (score > .4f) {
          cv::rectangle(frame, cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin),
            cv::Scalar(0, 0, 255), 3);
        }
      }
      cv::imshow("Detection", frame);
    }
    k = cv::waitKey(1);
  }
  interpreter.reset();
  tpu_context.reset();
  return 0;
}