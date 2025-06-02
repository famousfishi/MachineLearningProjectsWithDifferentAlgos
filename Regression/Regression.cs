using MachineLearningProjectsWithDifferentAlgos.Regression.DTOs;
using Microsoft.ML;

namespace MachineLearningProjectsWithDifferentAlgos.Regression
{
    public class Regression
    {
        public static void RunRegressionSampleForYearsOfExperience()
        {
            // Create a new MLContext
            MLContext mLContext = new();

            // Define the data schema
            List<InputModel> inputData = new()
            {
                new InputModel()
                {
                    YearsOfExperience = 1.1f,
                    Salary = 39343
                },
                new InputModel()
                {
                    YearsOfExperience = 2f,
                    Salary = 67909
                },
                new InputModel()
                {
                    YearsOfExperience = 3f,
                    Salary = 77000
                },
                new InputModel()
                {
                    YearsOfExperience = 4f,
                    Salary = 80000
                },
                new InputModel()
                {
                    YearsOfExperience = 5f,
                    Salary = 110000
                },
                new InputModel()
                {
                    YearsOfExperience = 6f,
                    Salary = 150000
                },
                new InputModel()
                {
                    YearsOfExperience = 7f,
                    Salary = 130000
                },
                new InputModel()
                {
                    YearsOfExperience = 8f,
                    Salary = 120000
                },
                new InputModel()
                {
                    YearsOfExperience = 9f,
                    Salary = 150000
                }
            };

            // Load data into IDataView
            IDataView trainingData = mLContext.Data.LoadFromEnumerable(inputData);

            // Define the training pipeline
            Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Data.RegressionPredictionTransformer<Microsoft.ML.Trainers.LinearRegressionModelParameters>> pipeline
                = mLContext.Transforms.Concatenate("Features", "YearsOfExperience")
                .Append(mLContext.Regression.Trainers.Sdca(labelColumnName: "Salary", maximumNumberOfIterations: 100));

            // Train the model
            Microsoft.ML.Data.TransformerChain<Microsoft.ML.Data.RegressionPredictionTransformer<Microsoft.ML.Trainers.LinearRegressionModelParameters>> model
                = pipeline.Fit(trainingData);

            // Create a prediction engine
            PredictionEngine<InputModel, OutputModel> predictionEngine = mLContext.Model.CreatePredictionEngine<InputModel, OutputModel>(model);

            // Create a new input instance for prediction
            InputModel inputInstance = new InputModel()
            {
                YearsOfExperience = 5.9f
            };

            // Make a prediction
            OutputModel prediction = predictionEngine.Predict(inputInstance);

            // Output the prediction result
            Console.WriteLine($"Predicted Salary for {inputInstance.YearsOfExperience} years of experience: {prediction.Salary:C2}");
        }
    }
}