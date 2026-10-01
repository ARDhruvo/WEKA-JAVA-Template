import java.util.Random;
import weka.associations.Apriori;
import weka.associations.FPGrowth;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class WekaUnifiedPipelineDemo {

    public static void main(String[] args) {
        try {
            // ================================================================
            // 1. DATA PREPROCESSING
            // ================================================================
            System.out.println(">>> 1. DATA PREPROCESSING");
            
            // Load dataset (e.g., standard Iris dataset containing numeric features)
            DataSource source = new DataSource("iris.arff");
            Instances rawData = source.getDataSet();
            if (rawData.classIndex() == -1) {
                rawData.setClassIndex(rawData.numAttributes() - 1);
            }

            // A. Clean missing values
            ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
            replaceMissing.setInputFormat(rawData);
            Instances cleanData = Filter.useFilter(rawData, replaceMissing);

            // B. Normalize numeric attributes (scale to [0, 1] for ML models)
            Normalize normalize = new Normalize();
            normalize.setInputFormat(cleanData);
            Instances normalizedData = Filter.useFilter(cleanData, normalize);

            // C. Discretize numeric attributes into nominal bins (required for Association Rules)
            Discretize discretize = new Discretize();
            discretize.setInputFormat(cleanData);
            Instances discretizedData = Filter.useFilter(cleanData, discretize);

            System.out.println("Preprocessed " + normalizedData.numInstances() + " instances successfully.\n");

            // ================================================================
            // 2. STANDARD MACHINE LEARNING (J48 Decision Tree)
            // ================================================================
            System.out.println(">>> 2. STANDARD MACHINE LEARNING (J48 Decision Tree)");
            
            J48 tree = new J48();
            tree.setOptions(new String[]{"-C", "0.25", "-M", "2"}); // Pruning confidence: 0.25, min instances: 2
            tree.buildClassifier(normalizedData);

            // 10-fold Cross-Validation
            Evaluation treeEval = new Evaluation(normalizedData);
            treeEval.crossValidateModel(tree, normalizedData, 10, new Random(1));
            System.out.printf("J48 Accuracy: %.2f%%\n", treeEval.pctCorrect());

            // Inference on the first record
            Instance sample = normalizedData.firstInstance();
            double classIdx = tree.classifyInstance(sample);
            System.out.println("Sample Prediction: " + normalizedData.classAttribute().value((int) classIdx) + "\n");

            // ================================================================
            // 3. ENSEMBLE LEARNING (AdaBoost with Decision Stumps)
            // ================================================================
            System.out.println(">>> 3. ENSEMBLE LEARNING (AdaBoostM1)");
            
            AdaBoostM1 ensemble = new AdaBoostM1();
            ensemble.setClassifier(new DecisionStump()); // Weak base learner
            ensemble.setNumIterations(25);              // 25 boosting iterations
            ensemble.buildClassifier(normalizedData);

            Evaluation ensembleEval = new Evaluation(normalizedData);
            ensembleEval.crossValidateModel(ensemble, normalizedData, 10, new Random(1));
            System.out.printf("AdaBoostM1 Accuracy: %.2f%%\n\n", ensembleEval.pctCorrect());

            // ================================================================
            // 4. ASSOCIATION RULE MINING (Apriori & FPGrowth)
            // ================================================================
            System.out.println(">>> 4. ASSOCIATION RULE MINING");
            
            // Unset class index (association rules require unsupervised transaction format)
            Instances associationData = new Instances(discretizedData);
            associationData.setClassIndex(-1);

            // A. Apriori
            Apriori apriori = new Apriori();
            apriori.setNumRules(5);
            apriori.setLowerBoundMinSupport(0.1); // 10% minimum support
            apriori.setMinMetric(0.8);            // 80% minimum confidence
            apriori.buildAssociations(associationData);
            System.out.println("--- Top Apriori Rules ---");
            System.out.println(apriori);

            // B. FPGrowth
            FPGrowth fpGrowth = new FPGrowth();
            fpGrowth.setNumRulesToFind(5);
            fpGrowth.setLowerBoundMinSupport(0.1);
            fpGrowth.setMinMetric(0.8);
            fpGrowth.buildAssociations(associationData);
            System.out.println("--- Top FPGrowth Rules ---");
            System.out.println(fpGrowth);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}