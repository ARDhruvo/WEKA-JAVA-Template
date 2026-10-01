import java.io.File;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.REPTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Weka standard classifier / regressor template.
 *
 * Compile: javac -cp ".;weka.jar" ClassifierTemplate.java   (use ':' on Linux/macOS)
 * Run:     java  -cp ".;weka.jar" ClassifierTemplate data/iris.arff [classIndex1Based]
 */
public class ClassifierTemplate {

    static Instances load(String path, int classIndex1Based) throws Exception {
        Instances data = new DataSource(path).getDataSet();
        if (classIndex1Based > 0) {
            data.setClassIndex(classIndex1Based - 1);
        } else if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    // ------------------------------------------------------------------
    // Evaluation helpers
    // ------------------------------------------------------------------

    static Evaluation crossValidate(Classifier c, Instances data, int folds, long seed) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(c, data, folds, new Random(seed)); // clones c internally; c itself stays untrained
        return eval;
    }

    static Evaluation holdOut(Classifier c, Instances train, Instances test) throws Exception {
        Classifier copy = AbstractClassifier.makeCopy(c);
        copy.buildClassifier(train);
        Evaluation eval = new Evaluation(train);
        eval.evaluateModel(copy, test);
        return eval;
    }

    static Instances[] split(Instances data, double trainPercent, long seed) {
        Instances copy = new Instances(data);
        copy.randomize(new Random(seed));
        int n = (int) Math.round(copy.numInstances() * trainPercent / 100.0);
        return new Instances[] { new Instances(copy, 0, n), new Instances(copy, n, copy.numInstances() - n) };
    }

    static void printClassification(Evaluation e) throws Exception {
        System.out.println(e.toSummaryString("\n=== Summary ===\n", false));
        System.out.println(e.toClassDetailsString());
        System.out.println(e.toMatrixString());
    }

    // ------------------------------------------------------------------
    // Model catalogue
    // ------------------------------------------------------------------

    static Map<String, Classifier> classificationModels() {
        Map<String, Classifier> m = new LinkedHashMap<>();

        m.put("ZeroR (baseline)", new ZeroR());

        OneR oneR = new OneR();
        oneR.setMinBucketSize(6);
        m.put("OneR", oneR);

        J48 j48 = new J48();
        j48.setConfidenceFactor(0.25f);
        j48.setMinNumObj(2);
        j48.setUnpruned(false);
        m.put("J48", j48);

        REPTree rep = new REPTree();
        rep.setMaxDepth(-1);
        rep.setMinNum(2.0);
        m.put("REPTree", rep);

        NaiveBayes nb = new NaiveBayes();
        nb.setUseKernelEstimator(false);
        nb.setUseSupervisedDiscretization(false);
        m.put("NaiveBayes", nb);

        m.put("IBk (k=3)", new IBk(3));

        SMO svmPoly = new SMO();
        svmPoly.setC(1.0);
        PolyKernel poly = new PolyKernel();
        poly.setExponent(1.0);
        svmPoly.setKernel(poly);
        m.put("SMO (poly)", svmPoly);

        SMO svmRbf = new SMO();
        svmRbf.setC(1.0);
        RBFKernel rbf = new RBFKernel();
        rbf.setGamma(0.01);
        svmRbf.setKernel(rbf);
        m.put("SMO (rbf)", svmRbf);

        Logistic logistic = new Logistic();
        logistic.setRidge(1.0E-8);
        m.put("Logistic", logistic);

        MultilayerPerceptron mlp = new MultilayerPerceptron();
        mlp.setHiddenLayers("a"); // "a" = (attributes + classes) / 2; "5,3" = two hidden layers
        mlp.setLearningRate(0.3);
        mlp.setMomentum(0.2);
        mlp.setTrainingTime(200);
        m.put("MultilayerPerceptron", mlp);

        JRip jrip = new JRip();
        jrip.setFolds(3);
        jrip.setMinNo(2.0);
        jrip.setOptimizations(2);
        m.put("JRip", jrip);

        m.put("PART", new PART());
        return m;
    }

    static Map<String, Classifier> regressionModels() {
        Map<String, Classifier> m = new LinkedHashMap<>();
        m.put("LinearRegression", new LinearRegression());
        m.put("M5P", new M5P());
        m.put("REPTree", new REPTree());
        return m;
    }

    // ------------------------------------------------------------------
    // Classification workflow
    // ------------------------------------------------------------------

    static void compareClassifiers(Instances data) throws Exception {
        System.out.printf("%-22s %8s %8s %8s %8s %8s%n", "Model", "Acc%", "Kappa", "wF1", "wAUC", "ms");
        for (Map.Entry<String, Classifier> entry : classificationModels().entrySet()) {
            long t0 = System.nanoTime();
            Evaluation e = crossValidate(entry.getValue(), data, 10, 1);
            long ms = (System.nanoTime() - t0) / 1_000_000;
            System.out.printf("%-22s %8.2f %8.3f %8.3f %8.3f %8d%n", entry.getKey(), e.pctCorrect(), e.kappa(),
                    e.weightedFMeasure(), e.weightedAreaUnderROC(), ms);
        }
    }

    static void inspectDecisionTree(Instances data) throws Exception {
        J48 tree = new J48();
        tree.buildClassifier(data);
        System.out.println(tree); // text tree
        System.out.printf("Leaves: %.0f  Tree size: %.0f%n", tree.measureNumLeaves(), tree.measureTreeSize());
        // tree.graph() returns the tree in Graphviz DOT format
    }

    static void predictDemo(Instances data) throws Exception {
        Instances[] s = split(data, 70, 7);
        J48 model = new J48();
        model.buildClassifier(s[0]);

        Instance x = (Instance) s[1].instance(0).copy();
        x.setClassMissing();
        double idx = model.classifyInstance(x);
        double[] dist = model.distributionForInstance(x);

        System.out.println("Predicted: " + data.classAttribute().value((int) idx));
        for (int i = 0; i < dist.length; i++) {
            System.out.printf("  P(%s) = %.3f%n", data.classAttribute().value(i), dist[i]);
        }
    }

    static void saveAndLoadModel(Instances data, String path) throws Exception {
        new File(path).getAbsoluteFile().getParentFile().mkdirs();
        NaiveBayes model = new NaiveBayes();
        model.buildClassifier(data);

        // model + empty header, so the attribute structure travels with the file
        SerializationHelper.writeAll(path, new Object[] { model, new Instances(data, 0) });

        Object[] loaded = SerializationHelper.readAll(path);
        Classifier restored = (Classifier) loaded[0];
        Instances header = (Instances) loaded[1];
        System.out.println("Restored " + restored.getClass().getSimpleName() + " with " + header.numAttributes()
                + " attributes");
    }

    static void optionStringsDemo() throws Exception {
        String[] opts = Utils.splitOptions("-C 0.25 -M 2");
        J48 j = new J48();
        j.setOptions(opts);
        System.out.println("J48 options now: " + Utils.joinOptions(j.getOptions()));

        Classifier byName = AbstractClassifier.forName("weka.classifiers.trees.J48", Utils.splitOptions("-U"));
        System.out.println("Created by class name: " + byName.getClass().getName());
    }

    static void runClassification(Instances data) throws Exception {
        Instances[] s = split(data, 70, 42);

        System.out.println("##### Hold-out: J48 #####");
        printClassification(holdOut(new J48(), s[0], s[1]));

        System.out.println("##### 10-fold CV: NaiveBayes #####");
        printClassification(crossValidate(new NaiveBayes(), data, 10, 1));

        System.out.println("##### Decision tree #####");
        inspectDecisionTree(data);

        System.out.println("##### Model comparison (10-fold CV) #####");
        compareClassifiers(data);

        System.out.println("##### Prediction #####");
        predictDemo(data);

        System.out.println("##### Persistence #####");
        saveAndLoadModel(data, "out/naivebayes.model");

        System.out.println("##### Options #####");
        optionStringsDemo();
    }

    // ------------------------------------------------------------------
    // Regression workflow (numeric class)
    // ------------------------------------------------------------------

    static void runRegression(Instances data) throws Exception {
        System.out.printf("%-22s %8s %10s %10s %8s%n", "Model", "Corr", "MAE", "RMSE", "RRSE%");
        for (Map.Entry<String, Classifier> entry : regressionModels().entrySet()) {
            Evaluation e = crossValidate(entry.getValue(), data, 10, 1);
            System.out.printf("%-22s %8.3f %10.3f %10.3f %8.2f%n", entry.getKey(), e.correlationCoefficient(),
                    e.meanAbsoluteError(), e.rootMeanSquaredError(), e.rootRelativeSquaredError());
        }
    }

    public static void main(String[] args) throws Exception {
        String path = args.length > 0 ? args[0] : "data/iris.arff";
        int classArg = args.length > 1 ? Integer.parseInt(args[1]) : -1;

        Instances data = load(path, classArg);
        if (data.classAttribute().isNominal()) {
            runClassification(data);
        } else {
            runRegression(data);
        }
    }
}
