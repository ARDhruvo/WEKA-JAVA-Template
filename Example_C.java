import java.util.Random;

import weka.associations.Apriori;
import weka.associations.AssociationRule;
import weka.associations.FPGrowth;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.REPTree;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * One program that uses all four parts of the Weka API.
 *
 * Compile: javac -cp ".;weka.jar" WekaBasicPipeline.java   (use ':' on Linux/macOS)
 * Run:     java  -cp ".;weka.jar" WekaBasicPipeline data/iris.arff
 */
public class WekaBasicPipeline {

    static Instances applyFilter(Instances data, Filter filter) throws Exception {
        filter.setInputFormat(data);
        return Filter.useFilter(data, filter);
    }

    static void evaluate(String name, Classifier model, Instances train, Instances test) throws Exception {
        model.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(model, test);
        System.out.printf("%-22s accuracy=%.2f%%  kappa=%.3f  weightedF1=%.3f%n", name, eval.pctCorrect(),
                eval.kappa(), eval.weightedFMeasure());
    }

    static void printRules(String title, Iterable<AssociationRule> rules) throws Exception {
        System.out.println("\n" + title);
        for (AssociationRule r : rules) {
            System.out.printf("  %s ==> %s  (%s=%.3f, support=%d)%n", r.getPremise(), r.getConsequence(),
                    r.getPrimaryMetricName(), r.getPrimaryMetricValue(), r.getTotalSupport());
        }
    }

    public static void main(String[] args) throws Exception {
        String path = args.length > 0 ? args[0] : "data/iris.arff";

        // 1. Load
        Instances data = new DataSource(path).getDataSet();
        data.setClassIndex(data.numAttributes() - 1);
        System.out.println("Loaded " + data.numInstances() + " instances, " + data.numAttributes() + " attributes");

        // 2. Preprocessing: shuffle, split, then fit filters on the training part only
        Instances shuffled = new Instances(data);
        shuffled.randomize(new Random(42));
        int trainSize = (int) Math.round(shuffled.numInstances() * 0.7);
        Instances train = new Instances(shuffled, 0, trainSize);
        Instances test = new Instances(shuffled, trainSize, shuffled.numInstances() - trainSize);

        ReplaceMissingValues missing = new ReplaceMissingValues();
        missing.setInputFormat(train);
        train = Filter.useFilter(train, missing);
        test = Filter.useFilter(test, missing);

        Normalize normalize = new Normalize();
        normalize.setInputFormat(train);
        train = Filter.useFilter(train, normalize);
        test = Filter.useFilter(test, normalize);

        // 3. Standard classifiers
        System.out.println("\n--- Standard classifiers ---");
        evaluate("J48", new J48(), train, test);
        evaluate("NaiveBayes", new NaiveBayes(), train, test);

        // 4. Ensembles
        System.out.println("\n--- Ensembles ---");
        Bagging bagging = new Bagging();
        bagging.setClassifier(new REPTree());
        bagging.setNumIterations(10);
        evaluate("Bagging", bagging, train, test);

        RandomForest forest = new RandomForest();
        forest.setOptions(Utils.splitOptions("-I 100 -K 0 -depth 0 -S 1"));
        evaluate("RandomForest", forest, train, test);

        AdaBoostM1 boost = new AdaBoostM1();
        boost.setClassifier(new DecisionStump());
        boost.setNumIterations(10);
        evaluate("AdaBoostM1", boost, train, test);

        // 5. Association rules: no class attribute, nominal data only
        Instances assoc = new Instances(data);
        assoc.setClassIndex(-1);

        Discretize discretize = new Discretize();
        discretize.setBins(3);
        assoc = applyFilter(assoc, discretize);

        Apriori apriori = new Apriori();
        apriori.setNumRules(5);
        apriori.setMinMetric(0.9);
        apriori.setLowerBoundMinSupport(0.1);
        apriori.buildAssociations(assoc);
        printRules("Apriori rules", apriori.getAssociationRules().getRules());

        NominalToBinary binarize = new NominalToBinary();
        binarize.setBinaryAttributesNominal(true);
        binarize.setTransformAllValues(true);
        Instances binary = applyFilter(assoc, binarize);

        FPGrowth fpGrowth = new FPGrowth();
        fpGrowth.setPositiveIndex(2);
        fpGrowth.setNumRulesToFind(5);
        fpGrowth.setMinMetric(0.9);
        fpGrowth.setLowerBoundMinSupport(0.1);
        fpGrowth.buildAssociations(binary);
        printRules("FPGrowth rules", fpGrowth.getAssociationRules().getRules());
    }
}
