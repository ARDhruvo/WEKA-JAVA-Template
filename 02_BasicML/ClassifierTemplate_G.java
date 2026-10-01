import java.util.Random;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class WekaStandardMLTemplate {

    public static void main(String[] args) {
        try {
            DataSource source = new DataSource("data/iris.arff");
            Instances data = source.getDataSet();
            data.setClassIndex(data.numAttributes() - 1);

            // --- Model 1: J48 Decision Tree (C4.5 implementation) ---
            J48 tree = new J48();
            String[] treeOptions = new String[]{"-C", "0.25", "-M", "2"}; // Confidence factor & min instances per leaf
            tree.setOptions(treeOptions);
            tree.buildClassifier(data);

            // Evaluate with 10-fold Cross-Validation
            Evaluation evalTree = new Evaluation(data);
            evalTree.crossValidateModel(tree, data, 10, new Random(1));
            System.out.println("=== J48 Evaluation ===");
            System.out.println("Accuracy: " + (evalTree.pctCorrect()) + "%");
            System.out.println(evalTree.toSummaryString());

            // --- Model 2: Naive Bayes ---
            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(data);

            Evaluation evalNB = new Evaluation(data);
            evalNB.crossValidateModel(nb, data, 10, new Random(1));
            System.out.println("=== Naive Bayes Evaluation ===");
            System.out.println("Accuracy: " + (evalNB.pctCorrect()) + "%");

            // --- Inference on a Single Instance ---
            Instance testInstance = data.firstInstance();
            double classIndex = tree.classifyInstance(testInstance);
            double[] probabilities = tree.distributionForInstance(testInstance);

            System.out.println("Predicted class: " + data.classAttribute().value((int) classIndex));
            System.out.println("Class probability distribution: " + java.util.Arrays.toString(probabilities));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}