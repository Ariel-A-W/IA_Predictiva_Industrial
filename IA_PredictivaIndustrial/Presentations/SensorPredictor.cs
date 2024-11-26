using Microsoft.ML.Data;

namespace IA_PredictivaIndustrial.Presentations;

public class SensorPredictor
{    
    [ColumnName("PredictedLabel")]
    public bool IsFaulty { get; set; }

    [ColumnName("Score")]
    public float Probabilidad { get; set; }
}
