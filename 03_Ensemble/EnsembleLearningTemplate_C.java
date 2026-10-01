import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.meta.RandomSubSpace;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.REPTree;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Weka ensemble learning template (nominal class).
 *
 * Compile: javac -cp ".;weka.jar" EnsembleLearningTemplate.java   (use ':' on Linux/macOS)
 * Run:     java  -cp ".;weka.jar" EnsembleLearningTemplate data/iris.arff [classIndex1Based]
 */
public class EnsembleLearningTemplate {

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
    // Ensemble builders
    // ------------------------------------------------------------------

    /** Bootstrap aggregation of unstable base learners. */
    static Bagging bagging(int iterations) {
        Bagging bag = new Bagging();
        bag.setClassifier(new REPTree());
        bag.setNumIterations(iterations);
        bag.setBagSizePercent(100);
        bag.setSeed(1);
        return bag;
    }

    /** Options string keeps this working across 3.8.x releases (-I trees, -K features, -depth, -S seed). */
    static RandomForest randomForest(int trees) throws Exception {
        RandomForest rf = new RandomForest();
        rf.setOptions(Utils.splitOptions("-I " + trees + " -K 0 -depth 0 -S 1"));
        return rf;
    }

    /** Sequential re-weighting; weak learner is a one-level tree. */
    static AdaBoostM1 adaBoost(int iterations) {
        AdaBoostM1 ada = new AdaBoostM1();
        ada.setClassifier(new DecisionStump());
        ada.setNumIterations(iterations);
        ada.setUseResampling(false);
        ada.setSeed(1);
        return ada;
    }

    static LogitBoost logitBoost(int iterations) {
        LogitBoost lb = new LogitBoost();
        lb.setClassifier(new DecisionStump());
        lb.setNumIterations(iterations);
        lb.setShrinkage(1.0);
        return lb;
    }

    /** Each tree sees a random subset of the attributes. */
    static RandomSubSpace randomSubSpace(int iterations) {
        RandomSubSpace rss = new RandomSubSpace();
        rss.setClassifier(new REPTree());
        rss.setSubSpaceSize(0.5);
        rss.setNumIterations(iterations);
        rss.setSeed(1);
        return rss;
    }

    /** Level-0 learners feed out-of-fold predictions to a level-1 meta learner. */
    static Stacking stacking() {
        Stacking st = new Stacking();
        st.setClassifiers(new Classifier[] { new J48(), new NaiveBayes(), new IBk(3) });
        st.setMetaClassifier(new Logistic());
        st.setNumFolds(10);
        return st;
    }

    /** Combine heterogeneous models with a fixed rule: average of probabilities, majority vote, product, min, max, median. */
    static Vote vote(int rule) {
        Vote v = new Vote();
        v.setClassifiers(new Classifier[] { new J48(), new NaiveBayes(), new IBk(3) });
        v.setCombinationRule(new SelectedTag(rule, Vote.TAGS_RULES));
        v.setSeed(1);
        return v;
    }

    // ------------------------------------------------------------------
    // Evaluation
    // ------------------------------------------------------------------

    static void cvLine(String name, Classifier c, Instances data) throws Exception {
        long t0 = System.nanoTime();
        Evaluation e = new Evaluation(data);
        e.crossValidateModel(c, data, 10, new Random(1));
        long ms = (System.nanoTime() - t0) / 1_000_000;
        System.out.printf("%-26s %8.2f %8.3f %8.3f %8.3f %8d%n", name, e.pctCorrect(), e.kappa(),
                e.weightedFMeasure(), e.weightedAreaUnderROC(), ms);
    }

    static void compareEnsembles(Instances data) throws Exception {
        Map<String, Classifier> models = new LinkedHashMap<>();
        models.put("J48 (single tree)", new J48());
        models.put("Bagging (REPTree)", bagging(10));
        models.put("RandomForest (100)", randomForest(100));
        models.put("AdaBoostM1 (stump)", adaBoost(10));
        models.put("LogitBoost (stump)", logitBoost(10));
        models.put("RandomSubSpace", randomSubSpace(10));
        models.put("Stacking (Logistic)", stacking());
        models.put("Vote (average)", vote(Vote.AVERAGE_RULE));
        models.put("Vote (majority)", vote(Vote.MAJORITY_VOTING_RULE));

        System.out.printf("%-26s %8s %8s %8s %8s %8s%n", "Model", "Acc%", "Kappa", "wF1", "wAUC", "ms");
        for (Map.Entry<String, Classifier> m : models.entrySet()) {
            cvLine(m.getKey(), m.getValue(), data);
        }
    }

    /** How accuracy changes as the ensemble grows. */
    static void sweepEnsembleSize(Instances data) throws Exception {
        System.out.println("Bagging accuracy vs number of iterations:");
        for (int n : new int[] { 1, 5, 10, 25, 50, 100 }) {
            Evaluation e = new Evaluation(data);
            e.crossValidateModel(bagging(n), data, 10, new Random(1));
            System.out.printf("  iterations=%-4d accuracy=%.2f%%%n", n, e.pctCorrect());
        }
    }

    public static void main(String[] args) throws Exception {
        String path = args.length > 0 ? args[0] : "data/iris.arff";
        int classArg = args.length > 1 ? Integer.parseInt(args[1]) : -1;

        Instances data = load(path, classArg);
        if (!data.classAttribute().isNominal()) {
            System.out.println("This template expects a nominal class. Bagging, RandomForest, Stacking and "
                    + "Vote (average rule) also accept numeric classes; boosting and majority voting do not.");
            return;
        }

        compareEnsembles(data);
        System.out.println();
        sweepEnsembleSize(data);
    }
}
