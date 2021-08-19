// Include dependencies
#include "sine_model_quantized.cc"
#include "tensorflow\lite\micro\all_ops_resolver.cc"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Setting up the test
TF_LITE_MICRO_TESTS_BEGIN
TF_LITE_MICRO_TEST(LoadModelAndPerformInference) {

// Getting ready to log data

    // Set up Logging
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

// Maping the model into usable data structure. This doesn't invovle any
//copying or parsing, it's a very lightweight operation
const tflite::Model* model = ::tflite::GetModel(g_sine_model_data);
if (model -> version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is shema version of %d not equal "
        "to supported version %d.\n",
        model->version(), TFLITE_SHEMA_VERSION
    );
    return 1;
}

// Creating AllOpsResolver
    // This pulls in all the operation implementations we need
tflite::ops::micro::AllOpsResolver resolver;

// Defining a Tensor Arena
    // Create an area of memory to use for input, output, and intermediate arraysconst int tensor_arena_size = 2 x 1024;
uint8_t tensor_arena[tensor_arena_size];

/*
/  Question: How large should our tensor be?
/  Finding the minimum value for model may require some trial and error
*/

// Creating an Interpreter
    // Build an interpreter to run the model with
tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, tensor_arena_size, error_reporter);

    // Allocate memory from the tensor_arena for the model's tensors
interpreter.AllocateTensors()

// Inspecting the Input Tensor
    // Obtain a pointer to the model 's input tensor
TfliteTensor* input = interpreter.input(0);

    // Make sure the input has the properties we expect
TF_LITE_MICRO_EXPECT_NE(nullptr, input);
    // The property "dims" tell us the tensor's shape. It has one element for
    // each dimension. Our input is a 2D tensor containing 1 element, so "dims"
    // should have size 2.
TF_LITE_MICRO_EXPECT_EQ(2, input->dims->size);
    // The input of each element gives the length of the corresponding tensor
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[1]);
    // The input is a 32 bit floating point value
TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, input->type);

// provide an input value
input->data.f[0] = 0;

// Run the model on this input and check that it succeeds
TfLiteStatus involke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed\n");
}
TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

// Reading the Output
TfLiteTensor* output = interpreter.output(0);

TF_LITE_MICRO_EXPECT_EQ(2, output->dims->size);
TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
TF_LITE_MICRO_EXPECT_EQ(2, input->dims->data[1]);
TF_LITE_MICRO_EXPECT_EQ(kTfLiteFloat32, output->type);


// Obtain the output value from the tensor
float value = output->data.f[0];

TF_LITE_MICRO_EXPECT_NEAR(0., value, 0.05);

// Running inference on several more values and confirm the expected value
input->data.f[0] = 1.;
interpreter.Invoke();
value = output->data.f[0];
TF_LITE_MICRO_EXPECT_NEAR(0.841, value, 0.05);

input->data.f[0] = 3.;
interpreter.Invoke();
value = output->data.f[0];
TF_LITE_MICRO_EXPECT_NEAR(0.141, value, 0.05);

input->data.f[0] = 5.;
interpreter.Invoke();
value = output->data.f[0];
TF_LITE_MICRO_EXPECT_NEAR(-0.959, value, 0.05);

}

TF_LITE_MICRO_TEST_END

