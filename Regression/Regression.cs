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
            var pipeline
                = mLContext.Transforms.Concatenate("Features", "YearsOfExperience")
                .Append(mLContext.Regression.Trainers.LbfgsPoissonRegression(labelColumnName: "Salary"));

            DataOperationsCatalog.TrainTestData split = mLContext.Data.TrainTestSplit(trainingData, testFraction: 0.8);

            // Train the model
            var model
                = pipeline.Fit(split.TrainSet);

            //evaluate the model
            IDataView predictions = model.Transform(split.TestSet);

            // Get the metrics
            Microsoft.ML.Data.RegressionMetrics metrics = mLContext.Regression.Evaluate(predictions, labelColumnName: "Salary");

            // Output the metrics
            Console.WriteLine($"R-squared: {metrics.RSquared}");
            Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
            Console.WriteLine($"Mean Squared Error: {metrics.MeanSquaredError}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
            Console.WriteLine($"Loss Function: {metrics.LossFunction}");

            //// Create a prediction engine
            //PredictionEngine<InputModel, OutputModel> predictionEngine = mLContext.Model.CreatePredictionEngine<InputModel, OutputModel>(model);

            //// Create a new input instance for prediction
            //InputModel inputInstance = new InputModel()
            //{
            //    YearsOfExperience = 5.9f
            //};

            //// Make a prediction
            //OutputModel prediction = predictionEngine.Predict(inputInstance);

            //// Output the prediction result
            //Console.WriteLine($"Predicted Salary for {inputInstance.YearsOfExperience} years of experience: {prediction.Salary:C2}");
        }
    }
}