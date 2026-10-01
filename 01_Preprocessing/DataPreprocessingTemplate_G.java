import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class WekaPreprocessingTemplate {

    public static void main(String[] args) {
        try {
            // 1. Load Dataset
            DataSource source = new DataSource("data/sample.arff");
            Instances dataset = source.getDataSet();

            // Set class index to the last attribute if not already set
            if (dataset.classIndex() == -1) {
                dataset.setClassIndex(dataset.numAttributes() - 1);
            }
            System.out.println("Original dataset size: " + dataset.numInstances() + " instances.");

            // 2. Handle Missing Values
            ReplaceMissingValues replaceMissing = new ReplaceMissingValues();
            replaceMissing.setInputFormat(dataset);
            Instances dataCleaned = Filter.useFilter(dataset, replaceMissing);

            // 3. Remove Specific Attributes (e.g., remove 1st attribute / ID column)
            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndices("1"); // 1-based indexing
            removeFilter.setInputFormat(dataCleaned);
            Instances dataReduced = Filter.useFilter(dataCleaned, removeFilter);

            // 4. Normalize Numeric Attributes (scales values to [0, 1])
            Normalize normalizeFilter = new Normalize();
            normalizeFilter.setScale(1.0);
            normalizeFilter.setTranslation(0.0);
            normalizeFilter.setInputFormat(dataReduced);
            Instances preprocessedData = Filter.useFilter(dataReduced, normalizeFilter);

            System.out.println("Preprocessed schema attributes: " + preprocessedData.numAttributes());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}