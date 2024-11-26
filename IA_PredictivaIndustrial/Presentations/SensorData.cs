using Microsoft.ML.Data;

namespace IA_PredictivaIndustrial.Presentations;

public class SensorData
{
    [LoadColumn(0)]
    public float Temperatura { get; set; }

    [LoadColumn(1)] 
    public float Vibracion { get; set; }

    [LoadColumn(2)]
    public float Humedad { get; set; }

    [LoadColumn(3), ColumnName("Label")]
    public bool IsFaulty { get; set; }
}
