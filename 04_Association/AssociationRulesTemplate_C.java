import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import weka.associations.Apriori;
import weka.associations.AssociationRule;
import weka.associations.AssociationRules;
import weka.associations.FPGrowth;
import weka.associations.Item;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

/**
 * Weka association rule mining template (Apriori and FPGrowth).
 *
 * Compile: javac -cp ".;weka.jar" AssociationRulesTemplate.java   (use ':' on Linux/macOS)
 * Run:     java  -cp ".;weka.jar" AssociationRulesTemplate data/weather.nominal.arff
 *
 * Association miners work on data WITHOUT a class attribute, so the class index is left unset.
 */
public class AssociationRulesTemplate {

    // ------------------------------------------------------------------
    // Loading and preparing data
    // ------------------------------------------------------------------

    static Instances load(String path) throws Exception {
        Instances data = new DataSource(path).getDataSet();
        data.setClassIndex(-1);
        return data;
    }

    /** Apriori only accepts nominal attributes; numeric ones are binned first. */
    static Instances toNominal(Instances data, int bins) throws Exception {
        weka.filters.unsupervised.attribute.Discretize d = new weka.filters.unsupervised.attribute.Discretize();
        d.setBins(bins);
        d.setUseEqualFrequency(true);
        d.setInputFormat(data);
        return Filter.useFilter(data, d);
    }

    /** FPGrowth wants two-valued attributes. This expands every nominal attribute into 0/1 nominal columns. */
    static Instances toBinaryNominal(Instances data) throws Exception {
        weka.filters.unsupervised.attribute.NominalToBinary f = new weka.filters.unsupervised.attribute.NominalToBinary();
        f.setBinaryAttributesNominal(true);
        f.setTransformAllValues(true);
        f.setInputFormat(data);
        return Filter.useFilter(data, f);
    }

    /** Builds a market-basket dataset: one nominal {f,t} attribute per item, one row per transaction. */
    static Instances buildTransactions(List<Set<String>> baskets) {
        Set<String> itemSet = new LinkedHashSet<>();
        for (Set<String> b : baskets) {
            itemSet.addAll(b);
        }
        List<String> items = new ArrayList<>(itemSet);

        ArrayList<String> values = new ArrayList<>(Arrays.asList("f", "t"));
        ArrayList<Attribute> attrs = new ArrayList<>();
        for (String item : items) {
            attrs.add(new Attribute(item, values));
        }

        Instances ds = new Instances("baskets", attrs, baskets.size());
        for (Set<String> basket : baskets) {
            double[] row = new double[items.size()];
            for (int i = 0; i < items.size(); i++) {
                row[i] = basket.contains(items.get(i)) ? 1 : 0; // index 1 = "t"
            }
            ds.add(new DenseInstance(1.0, row));
        }
        return ds;
    }

    // ------------------------------------------------------------------
    // Apriori
    // ------------------------------------------------------------------

    static Apriori runApriori(Instances data, int numRules, double minConfidence, double minSupport)
            throws Exception {
        Apriori apriori = new Apriori();
        apriori.setNumRules(numRules);
        apriori.setMinMetric(minConfidence);
        apriori.setLowerBoundMinSupport(minSupport);
        apriori.setUpperBoundMinSupport(1.0);
        apriori.setDelta(0.05); // support is lowered from the upper bound in steps of delta until numRules are found
        apriori.setRemoveAllMissingCols(true);
        apriori.buildAssociations(data);
        return apriori;
    }

    /** Rank by lift instead of confidence: metric type 1 = lift, minMetric then means minimum lift. */
    static Apriori runAprioriByLift(Instances data, int numRules, double minLift) throws Exception {
        Apriori apriori = new Apriori();
        apriori.setOptions(Utils.splitOptions("-N " + numRules + " -T 1 -C " + minLift + " -D 0.05 -U 1.0 -M 0.1"));
        apriori.buildAssociations(data);
        return apriori;
    }

    /** Market-basket mode: the first nominal value ("f") is treated as missing, so rules only mention bought items. */
    static Apriori runAprioriBasket(Instances baskets, int numRules, double minConfidence, double minSupport)
            throws Exception {
        Apriori apriori = new Apriori();
        apriori.setTreatZeroAsMissing(true);
        apriori.setNumRules(numRules);
        apriori.setMinMetric(minConfidence);
        apriori.setLowerBoundMinSupport(minSupport);
        apriori.buildAssociations(baskets);
        return apriori;
    }

    /** Class association rules: consequent restricted to the class attribute. Requires confidence as the metric. */
    static Apriori runClassAssociationRules(Instances data, int numRules) throws Exception {
        Apriori apriori = new Apriori();
        apriori.setCar(true);
        apriori.setClassIndex(-1); // -1 = last attribute
        apriori.setNumRules(numRules);
        apriori.setMinMetric(0.8);
        apriori.setLowerBoundMinSupport(0.1);
        apriori.buildAssociations(data);
        return apriori;
    }

    // ------------------------------------------------------------------
    // FPGrowth
    // ------------------------------------------------------------------

    static FPGrowth runFpGrowth(Instances binaryData, int numRules, double minConfidence, double minSupport)
            throws Exception {
        FPGrowth fp = new FPGrowth();
        fp.setPositiveIndex(2); // 2 = second nominal value ("t" or "1") counts as "item present"
        fp.setNumRulesToFind(numRules);
        fp.setMinMetric(minConfidence);
        fp.setLowerBoundMinSupport(minSupport);
        fp.setUpperBoundMinSupport(1.0);
        fp.setDelta(0.05);
        fp.setMaxNumberOfItems(-1); // -1 = no limit on rule length
        fp.buildAssociations(binaryData);
        return fp;
    }

    // ------------------------------------------------------------------
    // Working with rules
    // ------------------------------------------------------------------

    static String items(Collection<Item> items) {
        return items.stream().map(Item::toString).collect(Collectors.joining(" AND "));
    }

    static double lift(AssociationRule r) {
        try {
            return r.getNamedMetricValue("Lift");
        } catch (Exception e) {
            return Double.NaN;
        }
    }

    static void printRules(String title, AssociationRules rules) {
        System.out.println("=== " + title + " (" + rules.getRules().size() + " rules) ===");
        int i = 1;
        for (AssociationRule r : rules.getRules()) {
            System.out.printf("%2d. %s ==> %s%n    support=%d  %s=%.3f  lift=%.3f%n", i++, items(r.getPremise()),
                    items(r.getConsequence()), r.getTotalSupport(), r.getPrimaryMetricName(),
                    r.getPrimaryMetricValue(), lift(r));
        }
        System.out.println();
    }

    static List<AssociationRule> filterByLift(AssociationRules rules, double minLift) {
        List<AssociationRule> out = new ArrayList<>();
        for (AssociationRule r : rules.getRules()) {
            if (lift(r) >= minLift) {
                out.add(r);
            }
        }
        out.sort(Comparator.comparingDouble(AssociationRulesTemplate::lift).reversed());
        return out;
    }

    static void exportCsv(AssociationRules rules, String path) throws Exception {
        new File(path).getAbsoluteFile().getParentFile().mkdirs();
        try (PrintWriter w = new PrintWriter(path, "UTF-8")) {
            w.println("premise,consequence,support,confidence,lift,leverage,conviction");
            for (AssociationRule r : rules.getRules()) {
                w.printf("\"%s\",\"%s\",%d,%.4f,%.4f,%.4f,%.4f%n", items(r.getPremise()),
                        items(r.getConsequence()), r.getTotalSupport(), r.getNamedMetricValue("Confidence"),
                        r.getNamedMetricValue("Lift"), r.getNamedMetricValue("Leverage"),
                        r.getNamedMetricValue("Conviction"));
            }
        }
    }

    // ------------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        String path = args.length > 0 ? args[0] : "data/weather.nominal.arff";
        Instances data = load(path);

        Instances nominal = toNominal(data, 3);

        Apriori apriori = runApriori(nominal, 15, 0.8, 0.2);
        System.out.println(apriori); // Weka's own report: itemset counts per size and the best rules
        AssociationRules aprioriRules = apriori.getAssociationRules();
        printRules("Apriori (confidence >= 0.8)", aprioriRules);
        exportCsv(aprioriRules, "out/apriori_rules.csv");

        Apriori byLift = runAprioriByLift(nominal, 10, 1.2);
        printRules("Apriori ranked by lift", byLift.getAssociationRules());

        System.out.println("Rules with lift >= 1.5: " + filterByLift(aprioriRules, 1.5).size());

        Apriori car = runClassAssociationRules(nominal, 10);
        printRules("Class association rules", car.getAssociationRules());

        FPGrowth fp = runFpGrowth(toBinaryNominal(nominal), 15, 0.8, 0.2);
        printRules("FPGrowth (confidence >= 0.8)", fp.getAssociationRules());

        List<Set<String>> baskets = new ArrayList<>();
        baskets.add(new LinkedHashSet<>(Arrays.asList("bread", "milk", "eggs")));
        baskets.add(new LinkedHashSet<>(Arrays.asList("bread", "butter")));
        baskets.add(new LinkedHashSet<>(Arrays.asList("milk", "butter", "bread")));
        baskets.add(new LinkedHashSet<>(Arrays.asList("eggs", "milk")));
        baskets.add(new LinkedHashSet<>(Arrays.asList("bread", "milk", "butter", "eggs")));
        Instances tx = buildTransactions(baskets);

        printRules("Apriori on baskets", runAprioriBasket(tx, 10, 0.7, 0.4).getAssociationRules());
        printRules("FPGrowth on baskets", runFpGrowth(tx, 10, 0.7, 0.4).getAssociationRules());
    }
}
