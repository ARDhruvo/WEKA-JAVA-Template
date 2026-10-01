import weka.associations.Apriori;
import weka.associations.FPGrowth;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class WekaAssociationTemplate {

    public static void main(String[] args) {
        try {
            DataSource source = new DataSource("data/supermarket.arff");
            Instances data = source.getDataSet();

            // Association algorithms require categorical/nominal data
            NumericToNominal numToNom = new NumericToNominal();
            numToNom.setInputFormat(data);
            Instances nominalData = Filter.useFilter(data, numToNom);

            // --- 1. Apriori ---
            Apriori apriori = new Apriori();
            apriori.setNumRules(10);          // Max rules to extract
            apriori.setMinMetric(0.8);        // Minimum confidence (0.8 = 80%)
            apriori.setLowerBoundMinSupport(0.1); // Min support 10%
            apriori.buildAssociations(nominalData);

            System.out.println("=== Apriori Rules ===");
            System.out.println(apriori);

            // --- 2. FPGrowth (Frequent Pattern Tree) ---
            FPGrowth fpGrowth = new FPGrowth();
            fpGrowth.setNumRulesToFind(10);
            fpGrowth.setLowerBoundMinSupport(0.1);
            fpGrowth.setMinMetric(0.8);
            fpGrowth.buildAssociations(nominalData);

            System.out.println("=== FPGrowth Rules ===");
            System.out.println(fpGrowth);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}