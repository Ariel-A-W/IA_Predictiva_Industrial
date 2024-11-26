using IA_PredictivaIndustrial.Presentations;
using Microsoft.ML;
using Newtonsoft.Json;

namespace IA_PredictivaIndustrial.Process;

public class IndustrialControlProcess
{
    public void Process()
    {
        var mlContext = new MLContext();

        string relativePath = Path.Combine("Data", "sensor.json");
        string filePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, relativePath);

        if (!File.Exists(filePath))
        {
            Console.WriteLine("Archivo JSON no encontrado.");
        }
        else
        {
            string jsonContent = File.ReadAllText(filePath);

            var objectContent = JsonConvert.DeserializeObject<List<SensorData>>(jsonContent);

            var lstSensor = new List<SensorData>();
            foreach (var item in objectContent!)
            {
                lstSensor.Add(item);
            }

            var positiveSample = lstSensor.Where(p => p.IsFaulty == true).ToList();

            // Balancear 
            //var balancedData = lstSensor.Concat(lstSensor.Where(x => x.IsFaulty == true)).ToList();
            var balancedData = lstSensor.Concat(positiveSample).ToList();

            // Cargar datos
            var data = mlContext.Data.LoadFromEnumerable<SensorData>(balancedData);

            // Dividir datos en entrenamiento y prueba
            var trainTestSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // Configurar pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", "Temperatura", "Vibracion", "Humedad")
                .Append(mlContext.Transforms.NormalizeMinMax("Temperatura"))
                .Append(mlContext.Transforms.NormalizeMinMax("Vibracion"))
                .Append(mlContext.Transforms.NormalizeMinMax("Humedad"))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression());

            // Entrenar modelo
            var model = pipeline.Fit(trainTestSplit.TrainSet);

            // Realizar predicciones
            var predictions = model.Transform(trainTestSplit.TestSet);            

            var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "PredictedLabel", scoreColumnName: "Score");

            // Usar modelo para predicciones en tiempo real
            var predictor = mlContext.Model.CreatePredictionEngine<SensorData, SensorPredictor>(model);

            var samplesTest = new List<SensorData>();

            // Cargar algunas muestras.
            samplesTest.AddRange([
                new SensorData
                    {
                        Temperatura = 870,
                        Vibracion = 1.5F,
                        Humedad = 60
                    },
                    new SensorData
                    {
                        Temperatura = 790,
                        Vibracion = 0.5F,
                        Humedad = 40
                    }
            ]);

            if (samplesTest == null || samplesTest.Count == 0)
            {
                Console.WriteLine("Se requiere del uso de parámetros para la evaluación.");
            }
            else
            {
                // Cajetilla de muestra de resultados.
                var evaCountInt = 1;
                Console.WriteLine("Evaluación Predictiva Resultante");
                Console.WriteLine("********************************\n");
                foreach (var item in samplesTest)
                {
                    var result = predictor.Predict(item);
                    // Conversión de Kelvin a Centígrados
                    float tempCelsius = item.Temperatura - 273.15f;
                    float tempFaranheit =  (9/5 * (item.Temperatura - 273.15f) + 32);
                    Console.WriteLine("Datos:");
                    Console.WriteLine(
                        $"Temperaturas:\n" +
                        $"K° {item.Temperatura:F2}\n" +
                        $"C° {tempCelsius:F2}\n" +
                        $"F° {tempFaranheit:F2}");
                    Console.WriteLine($"Vibración: {item.Vibracion:F2} mm/s");
                    Console.WriteLine($"Humedad: {item.Humedad:F2}%\n");
                    Console.WriteLine($"Muestra para Evaluación: {evaCountInt}\n");
                    Console.WriteLine($"Precisión: {metrics.Accuracy:F2}");
                    Console.WriteLine($"Probabilidad de Falla: {result.Probabilidad:F2}%");
                    var status = result.IsFaulty == true ? "Falla" : "Correcto";
                    Console.WriteLine($"Estado: {status} [{result.IsFaulty}]");
                    evaCountInt++;
                }
            }
        }        
    }
}
