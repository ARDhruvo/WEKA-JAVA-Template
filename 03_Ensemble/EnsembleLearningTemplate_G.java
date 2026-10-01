import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaEnsembleTemplate {

    public static void main(String[] args) {
        try {
            DataSource source = new DataSource("data/diabetes.arff");
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            // --- 1. Random Forest (Bagging + Feature Subsampling) ---
            RandomForest rf = new RandomForest();
            rf.setNumIterations(100); // Number of trees
            rf.setNumExecutionSlots(0); // 0 = automatic multi-threading based on CPU cores
            rf.setSeed(42);
            rf.buildClassifier(data);

            Evaluation evalRF = new Evaluation(data);
            evalRF.crossValidateModel(rf, data, 10, new Random(1));
            System.out.println("Random Forest Accuracy: " + evalRF.pctCorrect() + "%");

            // --- 2. AdaBoostM1 (Adaptive Boosting) ---
            AdaBoostM1 boost = new AdaBoostM1();
            boost.setClassifier(new DecisionStump()); // Base learner
            boost.setNumIterations(50);              // Number of boosting iterations
            boost.setWeightThreshold(100);           // Percentage of weight mass
            boost.buildClassifier(data);

            Evaluation evalBoost = new Evaluation(data);
            evalBoost.crossValidateModel(boost, data, 10, new Random(1));
            System.out.println("AdaBoostM1 Accuracy: " + evalBoost.pctCorrect() + "%");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}