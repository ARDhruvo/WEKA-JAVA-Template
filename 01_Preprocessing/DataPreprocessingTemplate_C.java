import java.io.File;
import java.util.Arrays;
import java.util.Random;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.PrincipalComponents;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * Weka data preprocessing template.
 *
 * Compile: javac -cp ".;weka.jar" DataPreprocessingTemplate.java   (use ':' on Linux/macOS)
 * Run:     java  -cp ".;weka.jar" DataPreprocessingTemplate data/iris.arff [classIndex1Based]
 */
public class DataPreprocessingTemplate {

    // ------------------------------------------------------------------
    // Loading, inspecting, saving
    // ------------------------------------------------------------------

    static Instances load(String path, int classIndex1Based) throws Exception {
        DataSource source = new DataSource(path); // .arff, .csv, .json, .xrff ...
        Instances data = source.getDataSet();
        if (classIndex1Based > 0) {
            data.setClassIndex(classIndex1Based - 1);
        } else if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }

    static void describe(Instances data) throws Exception {
        System.out.println("Relation: " + data.relationName());
        System.out.println("Instances: " + data.numInstances() + "  Attributes: " + data.numAttributes());
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute a = data.attribute(i);
            String type = a.isNumeric() ? "numeric" : a.isNominal() ? "nominal" : a.isString() ? "string" : "other";
            int missing = data.attributeStats(i).missingCount;
            System.out.printf("  [%d] %-20s %-8s missing=%d%s%n", i, a.name(), type, missing,
                    i == data.classIndex() ? "  <-- class" : "");
        }
    }

    static void brief(String label, Instances d) {
        System.out.println(label + ": " + d.numInstances() + " instances, " + d.numAttributes() + " attributes");
    }

    static void saveArff(Instances data, String path) throws Exception {
        new File(path).getAbsoluteFile().getParentFile().mkdirs();
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(path));
        saver.writeBatch();
    }

    static void saveCsv(Instances data, String path) throws Exception {
        new File(path).getAbsoluteFile().getParentFile().mkdirs();
        CSVSaver saver = new CSVSaver();
        saver.setInstances(data);
        saver.setFile(new File(path));
        saver.writeBatch();
    }

    // ------------------------------------------------------------------
    // Generic filter application
    // ------------------------------------------------------------------

    static Instances applyFilter(Instances data, Filter filter) throws Exception {
        filter.setInputFormat(data); // must be called AFTER the class index is set and options are configured
        return Filter.useFilter(data, filter);
    }

    // ------------------------------------------------------------------
    // Attribute filters
    // ------------------------------------------------------------------

    /** indices are 1-based, e.g. "1,3-5,last". */
    static Instances removeAttributes(Instances data, String indices) throws Exception {
        Remove remove = new Remove();
        remove.setAttributeIndices(indices);
        remove.setInvertSelection(false); // true keeps only the listed attributes
        return applyFilter(data, remove);
    }

    /** Numeric -> mean, nominal -> mode. */
    static Instances replaceMissing(Instances data) throws Exception {
        return applyFilter(data, new ReplaceMissingValues());
    }

    /** Min-max scaling to [0,1]. */
    static Instances normalize(Instances data) throws Exception {
        Normalize norm = new Normalize();
        norm.setScale(1.0);
        norm.setTranslation(0.0);
        return applyFilter(data, norm);
    }

    /** Z-score: zero mean, unit variance. */
    static Instances standardize(Instances data) throws Exception {
        return applyFilter(data, new Standardize());
    }

    static Instances discretize(Instances data, int bins, boolean equalFrequency) throws Exception {
        weka.filters.unsupervised.attribute.Discretize d = new weka.filters.unsupervised.attribute.Discretize();
        d.setBins(bins);
        d.setUseEqualFrequency(equalFrequency);
        d.setAttributeIndices("first-last");
        return applyFilter(data, d);
    }

    /** Entropy/MDL based. Needs a nominal class. */
    static Instances discretizeSupervised(Instances data) throws Exception {
        weka.filters.supervised.attribute.Discretize d = new weka.filters.supervised.attribute.Discretize();
        d.setUseBetterEncoding(true);
        return applyFilter(data, d);
    }

    /** Turn numeric codes into nominal values, e.g. range "1" or "last". */
    static Instances numericToNominal(Instances data, String range) throws Exception {
        weka.filters.unsupervised.attribute.NumericToNominal f = new weka.filters.unsupervised.attribute.NumericToNominal();
        f.setAttributeIndices(range);
        return applyFilter(data, f);
    }

    /** One-hot style encoding for nominal attributes with more than two values. */
    static Instances nominalToBinary(Instances data) throws Exception {
        weka.filters.unsupervised.attribute.NominalToBinary f = new weka.filters.unsupervised.attribute.NominalToBinary();
        f.setBinaryAttributesNominal(false);
        return applyFilter(data, f);
    }

    /** For datasets with a string attribute holding raw text. */
    static Instances textToWordVector(Instances data) throws Exception {
        weka.filters.unsupervised.attribute.StringToWordVector f = new weka.filters.unsupervised.attribute.StringToWordVector();
        f.setWordsToKeep(1000);
        f.setLowerCaseTokens(true);
        f.setOutputWordCounts(true);
        f.setTFTransform(true);
        f.setIDFTransform(true);
        return applyFilter(data, f);
    }

    // ------------------------------------------------------------------
    // Instance filters and splitting
    // ------------------------------------------------------------------

    /** Biases the sample towards a uniform class distribution (1.0 = fully balanced). */
    static Instances balanceClasses(Instances data) throws Exception {
        weka.filters.supervised.instance.Resample rs = new weka.filters.supervised.instance.Resample();
        rs.setBiasToUniformClass(1.0);
        rs.setSampleSizePercent(100.0);
        rs.setNoReplacement(false);
        rs.setRandomSeed(1);
        return applyFilter(data, rs);
    }

    static Instances shuffle(Instances data, long seed) {
        Instances copy = new Instances(data);
        copy.randomize(new Random(seed));
        return copy;
    }

    static Instances[] trainTestSplit(Instances data, double trainPercent, long seed) {
        Instances copy = shuffle(data, seed);
        int trainSize = (int) Math.round(copy.numInstances() * trainPercent / 100.0);
        Instances train = new Instances(copy, 0, trainSize);
        Instances test = new Instances(copy, trainSize, copy.numInstances() - trainSize);
        return new Instances[] { train, test };
    }

    /** Folds for manual cross-validation. */
    static void foldsDemo(Instances data, int folds, long seed) {
        Instances copy = shuffle(data, seed);
        if (copy.classAttribute().isNominal()) {
            copy.stratify(folds);
        }
        for (int i = 0; i < folds; i++) {
            Instances train = copy.trainCV(folds, i);
            Instances test = copy.testCV(folds, i);
            System.out.println("fold " + (i + 1) + ": train=" + train.numInstances() + " test=" + test.numInstances());
        }
    }

    /** Fit the filter on train only, then reuse it on test so both share one header. */
    static Instances[] standardizeTrainTest(Instances train, Instances test) throws Exception {
        Standardize f = new Standardize();
        f.setInputFormat(train);
        Instances trainF = Filter.useFilter(train, f);
        Instances testF = Filter.useFilter(test, f);
        return new Instances[] { trainF, testF };
    }

    // ------------------------------------------------------------------
    // Attribute selection and dimensionality reduction
    // ------------------------------------------------------------------

    static Instances selectWithCfs(Instances data) throws Exception {
        AttributeSelection sel = new AttributeSelection();
        sel.setEvaluator(new CfsSubsetEval());
        sel.setSearch(new BestFirst());
        sel.SelectAttributes(data);
        System.out.println("CFS selected (0-based, class last): " + Arrays.toString(sel.selectedAttributes()));
        return sel.reduceDimensionality(data);
    }

    static void rankWithInfoGain(Instances data, int topN) throws Exception {
        Ranker ranker = new Ranker();
        ranker.setNumToSelect(topN);
        AttributeSelection sel = new AttributeSelection();
        sel.setEvaluator(new InfoGainAttributeEval());
        sel.setSearch(ranker);
        sel.SelectAttributes(data);
        for (double[] r : sel.rankedAttributes()) {
            System.out.printf("  %-20s infoGain=%.4f%n", data.attribute((int) r[0]).name(), r[1]);
        }
    }

    static Instances pca(Instances data, double varianceCovered) throws Exception {
        PrincipalComponents pc = new PrincipalComponents();
        pc.setVarianceCovered(varianceCovered);
        weka.filters.supervised.attribute.AttributeSelection f = new weka.filters.supervised.attribute.AttributeSelection();
        f.setEvaluator(pc);
        f.setSearch(new Ranker());
        return applyFilter(data, f);
    }

    // ------------------------------------------------------------------
    // Chaining
    // ------------------------------------------------------------------

    /** Several filters in a fixed order, used as a single filter. */
    static Instances chain(Instances data) throws Exception {
        MultiFilter multi = new MultiFilter();
        multi.setFilters(new Filter[] { new ReplaceMissingValues(), new Normalize() });
        return applyFilter(data, multi); // do not call setInputFormat on the inner filters
    }

    /** Filter + classifier in one object: the filter is refit on each training fold, so no leakage. */
    static void filteredClassifierCv(Instances data) throws Exception {
        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(new Standardize());
        fc.setClassifier(new J48());
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(fc, data, 10, new Random(1));
        System.out.printf("FilteredClassifier(Standardize + J48) 10-fold accuracy: %.2f%%%n", eval.pctCorrect());
    }

    // ------------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        String path = args.length > 0 ? args[0] : "data/iris.arff";
        int classArg = args.length > 1 ? Integer.parseInt(args[1]) : -1;

        Instances data = load(path, classArg);
        describe(data);

        brief("Remove attribute 1", removeAttributes(data, "1"));
        brief("Replace missing", replaceMissing(data));
        brief("Normalize", normalize(data));
        brief("Standardize", standardize(data));
        brief("Discretize (5 bins)", discretize(data, 5, false));
        brief("Numeric->nominal (attr 1)", numericToNominal(data, "1"));
        brief("Nominal->binary", nominalToBinary(data));

        if (data.checkForStringAttributes()) {
            brief("Word vector", textToWordVector(data));
        }

        if (data.classAttribute().isNominal()) {
            brief("Supervised discretize", discretizeSupervised(data));
            brief("Balanced resample", balanceClasses(data));
            brief("CFS subset", selectWithCfs(data));
            rankWithInfoGain(data, 3);
            filteredClassifierCv(data);
        }

        brief("PCA (95% variance)", pca(data, 0.95));
        brief("MultiFilter chain", chain(data));

        Instances[] split = trainTestSplit(data, 70.0, 42);
        brief("Train", split[0]);
        brief("Test", split[1]);
        Instances[] scaled = standardizeTrainTest(split[0], split[1]);
        brief("Train (standardized)", scaled[0]);
        brief("Test (standardized)", scaled[1]);

        foldsDemo(data, 5, 1);

        saveArff(normalize(data), "out/preprocessed.arff");
        saveCsv(normalize(data), "out/preprocessed.csv");
        System.out.println("Saved out/preprocessed.arff and out/preprocessed.csv");
    }
}
