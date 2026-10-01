# Weka JAR API Documentation

Reference for using `weka.jar` (Weka 3.8.x) from Java code. It covers data preprocessing, standard classifiers, ensemble learning and association rule mining, and matches the four template files in this folder:

| Area | Template file |
|---|---|
| 1. Data preprocessing | `DataPreprocessingTemplate.java` |
| 2. Standard machine learning | `ClassifierTemplate.java` |
| 3. Ensemble learning | `EnsembleLearningTemplate.java` |
| 4. Association rule mining | `AssociationRulesTemplate.java` |

Parameter names and defaults below follow Weka 3.8.x. Check the Javadoc of the exact `weka.jar` you are using if a method is reported as missing.

---

## 0. Setup and core concepts

### 0.1 Getting the JAR

- Take `weka.jar` from your Weka installation folder, or
- Maven: `nz.ac.waikato.cms.weka:weka-stable:3.8.6` (any 3.8.x works for these templates).

Sample datasets (`iris.arff`, `weather.nominal.arff`, `diabetes.arff`, `supermarket.arff`, `cpu.arff` ...) live in the `data/` folder of the Weka installation.

### 0.2 Compile and run

| OS | Compile | Run |
|---|---|---|
| Windows | `javac -cp ".;weka.jar" Name.java` | `java -cp ".;weka.jar" Name data/iris.arff` |
| Linux / macOS | `javac -cp ".:weka.jar" Name.java` | `java -cp ".:weka.jar" Name data/iris.arff` |

In an IDE (IntelliJ, Eclipse, VS Code), add `weka.jar` as a library to the project.

### 0.3 Core classes

| Class | Package | Role |
|---|---|---|
| `Instances` | `weka.core` | A dataset: header (attributes) plus rows |
| `Instance` / `DenseInstance` | `weka.core` | One row |
| `Attribute` | `weka.core` | Column definition (numeric, nominal, string, date) |
| `DataSource` | `weka.core.converters.ConverterUtils` | Loads ARFF, CSV, JSON, XRFF and more by file extension |
| `ArffSaver`, `CSVSaver` | `weka.core.converters` | Write datasets |
| `Filter` | `weka.filters` | Base class of all preprocessing filters |
| `Classifier`, `AbstractClassifier` | `weka.classifiers` | Base of all classifiers and regressors |
| `Evaluation` | `weka.classifiers` | Cross-validation, hold-out, metrics |
| `Utils` | `weka.core` | Option string helpers (`splitOptions`, `joinOptions`) |
| `SerializationHelper` | `weka.core` | Save and load models |
| `SelectedTag` | `weka.core` | Enum-like option value (used by `Vote`, some metric settings) |

### 0.4 Rules that apply everywhere

1. **Set the class index** before using supervised algorithms: `data.setClassIndex(data.numAttributes() - 1)`. Unset means `-1`, which causes "Class index is negative (not set)!".
2. **Index bases differ.** Java methods on `Instances` (`attribute(i)`, `setClassIndex`) are 0-based. Filter range strings (`"1,3-5,last"`) and command-line options are 1-based.
3. **Configure, then initialise.** For filters, set all options before `setInputFormat(data)`.
4. **Every algorithm has an options string** equivalent to its setters, e.g. `Utils.splitOptions("-C 0.25 -M 2")` then `setOptions(...)`. `getOptions()` returns the current configuration.
5. **`Instances` copies are cheap but not implicit.** `new Instances(data)` makes a copy; filters return new `Instances` and never modify the input.
6. **Most methods throw `Exception`.** Declare `throws Exception` on `main` for experimentation.

---

## 1. Data preprocessing

### 1.1 The filter pattern

```java
Filter f = new weka.filters.unsupervised.attribute.Normalize();
f.setInputFormat(data);                    // 1. tell the filter the input structure
Instances out = Filter.useFilter(data, f); // 2. apply
```

Naming convention: `weka.filters.<supervised|unsupervised>.<attribute|instance>.<Name>`

- **supervised** filters use the class attribute (class index must be set, usually nominal).
- **unsupervised** filters ignore the class (most leave it untouched).
- **attribute** filters change columns; **instance** filters change rows.

### 1.2 Attribute filters

| Filter | Purpose | Key setters |
|---|---|---|
| `unsupervised.attribute.Remove` | Delete or keep columns | `setAttributeIndices("1,3-5,last")`, `setInvertSelection(boolean)` |
| `unsupervised.attribute.ReplaceMissingValues` | Fill missing values with mean (numeric) or mode (nominal) | none |
| `unsupervised.attribute.Normalize` | Scale numeric attributes, default to [0,1] | `setScale(double)`, `setTranslation(double)` |
| `unsupervised.attribute.Standardize` | Zero mean, unit variance | none |
| `unsupervised.attribute.Discretize` | Bin numeric attributes | `setBins(int)`, `setUseEqualFrequency(boolean)`, `setAttributeIndices(String)` |
| `supervised.attribute.Discretize` | Class-aware (MDL) binning | `setUseBetterEncoding(boolean)`, `setUseKononenko(boolean)` |
| `unsupervised.attribute.NumericToNominal` | Treat numeric codes as categories | `setAttributeIndices(String)` |
| `unsupervised.attribute.NominalToBinary` | Expand nominal to 0/1 columns | `setBinaryAttributesNominal(boolean)`, `setTransformAllValues(boolean)` |
| `unsupervised.attribute.StringToWordVector` | Text to bag-of-words | `setWordsToKeep(int)`, `setLowerCaseTokens(boolean)`, `setOutputWordCounts(boolean)`, `setTFTransform(boolean)`, `setIDFTransform(boolean)`, `setTokenizer(Tokenizer)`, `setStopwordsHandler(StopwordsHandler)` |
| `supervised.attribute.AttributeSelection` | Feature selection as a filter | `setEvaluator(ASEvaluation)`, `setSearch(ASSearch)` |

Notes:

- `Discretize` skips the class attribute when the class index is set before `setInputFormat`.
- `StringToWordVector` rearranges attributes in its output; check `out.classIndex()` afterwards.
- `Normalize` and `Standardize` learn their statistics in `setInputFormat`/first batch. See 1.5 for train/test use.

### 1.3 Instance filters

| Filter | Purpose | Key setters |
|---|---|---|
| `supervised.instance.Resample` | Resample with optional class balancing | `setBiasToUniformClass(double 0..1)`, `setSampleSizePercent(double)`, `setNoReplacement(boolean)`, `setRandomSeed(int)` |
| `unsupervised.instance.Resample` | Plain random sampling | `setSampleSizePercent`, `setNoReplacement`, `setRandomSeed` |
| `unsupervised.instance.Randomize` | Shuffle rows | `setRandomSeed(int)` |
| `unsupervised.instance.RemovePercentage` | Drop a percentage of rows (hold-out split) | `setPercentage(double)`, `setInvertSelection(boolean)` |
| `unsupervised.instance.RemoveWithValues` | Drop rows by attribute value | `setAttributeIndex(String)`, `setNominalIndices(String)`, `setSplitPoint(double)` |

SMOTE is not in the core JAR. It is distributed as the `SMOTE` package through Weka's package manager.

### 1.4 Attribute selection without a filter

```java
AttributeSelection sel = new AttributeSelection();
sel.setEvaluator(new CfsSubsetEval());
sel.setSearch(new BestFirst());
sel.SelectAttributes(data);
int[] chosen = sel.selectedAttributes();      // 0-based, class index is the last entry
Instances reduced = sel.reduceDimensionality(data);
```

| Evaluator | Pairs with | Result |
|---|---|---|
| `CfsSubsetEval` | `BestFirst`, `GreedyStepwise` | A subset of attributes |
| `InfoGainAttributeEval`, `GainRatioAttributeEval`, `ReliefFAttributeEval` | `Ranker` | A ranking (`sel.rankedAttributes()` returns `double[][]` of index and score) |
| `PrincipalComponents` | `Ranker` | New component attributes (`setVarianceCovered(0.95)`) |

`Ranker.setNumToSelect(n)` keeps the top n attributes.

### 1.5 Splitting, folds, and avoiding leakage

```java
Instances copy = new Instances(data);
copy.randomize(new Random(42));
int trainSize = (int) Math.round(copy.numInstances() * 0.7);
Instances train = new Instances(copy, 0, trainSize);
Instances test  = new Instances(copy, trainSize, copy.numInstances() - trainSize);
```

- `data.stratify(k)` then `data.trainCV(k, i)` / `data.testCV(k, i)` produce stratified folds.
- **Batch filtering.** Fit the filter on the training set, then apply the same filter object to the test set. Calling `setInputFormat` again on the test data refits the filter and leaks information.
- **`FilteredClassifier`** wraps a filter and a classifier and refits the filter inside every cross-validation fold. Prefer it whenever the filter learns statistics (normalise, discretise, select attributes).
- **`MultiFilter`** chains filters in order and acts as one filter. Call `setInputFormat` only on the `MultiFilter`.

### 1.6 Loading and saving

| Task | Code |
|---|---|
| Load | `new DataSource(path).getDataSet()` |
| Save ARFF | `ArffSaver s = new ArffSaver(); s.setInstances(d); s.setFile(new File(p)); s.writeBatch();` |
| Save CSV | same with `CSVSaver` |
| Summary | `data.toSummaryString()`, `data.attributeStats(i)` (fields `missingCount`, `distinctCount`, `nominalCounts`, `numericStats`) |

---

## 2. Standard machine learning

### 2.1 Classifier lifecycle

```java
Classifier c = new J48();
c.buildClassifier(train);                       // train
double idx    = c.classifyInstance(inst);       // predicted class index (or numeric value for regression)
double[] dist = c.distributionForInstance(inst);// class probabilities (nominal class only)
```

Label of a prediction: `data.classAttribute().value((int) idx)`.

### 2.2 Common algorithms

| Algorithm | Class | Key setters (default) | Notes |
|---|---|---|---|
| Baseline | `rules.ZeroR` | none | Majority class (or mean). Always report it as the reference |
| One rule | `rules.OneR` | `setMinBucketSize(6)` | Single attribute rule |
| Decision tree (C4.5) | `trees.J48` | `setConfidenceFactor(0.25f)`, `setMinNumObj(2)`, `setUnpruned(false)`, `setBinarySplits`, `setReducedErrorPruning` | `toString()` prints the tree, `graph()` returns DOT, `measureNumLeaves()`, `measureTreeSize()` |
| Fast tree | `trees.REPTree` | `setMaxDepth(-1)`, `setMinNum(2.0)`, `setNumFolds(3)` | Handles numeric class too |
| Stump | `trees.DecisionStump` | none | One-level tree, typical boosting base learner |
| Naive Bayes | `bayes.NaiveBayes` | `setUseKernelEstimator(false)`, `setUseSupervisedDiscretization(false)` | Gaussian per class for numeric attributes by default |
| Bayesian network | `bayes.BayesNet` | `setEstimator`, `setSearchAlgorithm` | Structure learning options |
| k-NN | `lazy.IBk` | constructor `new IBk(k)`, `setKNN(1)`, `setCrossValidate(false)`, `setDistanceWeighting(new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING))` | Normalises distances by default |
| SVM | `functions.SMO` | `setC(1.0)`, `setKernel(Kernel)`, `setBuildCalibrationModels` | Kernels in `functions.supportVector`: `PolyKernel` (`setExponent`), `RBFKernel` (`setGamma`), `NormalizedPolyKernel` |
| Logistic regression | `functions.Logistic` | `setRidge(1.0E-8)`, `setMaxIts(-1)` | Multinomial |
| Neural network | `functions.MultilayerPerceptron` | `setHiddenLayers("a")`, `setLearningRate(0.3)`, `setMomentum(0.2)`, `setTrainingTime(500)`, `setNormalizeAttributes(true)` | `"a"` = (attributes + classes)/2, `"5,3"` = two hidden layers |
| Rule learners | `rules.JRip`, `rules.PART` | JRip: `setFolds(3)`, `setMinNo(2.0)`, `setOptimizations(2)` | Readable rule sets |
| Linear regression | `functions.LinearRegression` | `setEliminateColinearAttributes(true)`, `setRidge` | Numeric class |
| Model tree | `trees.M5P` | `setMinNumInstances(4.0)`, `setUnpruned` | Numeric class |

Packages `weka.classifiers.*` contain: `bayes`, `functions`, `lazy`, `meta`, `rules`, `trees`, `misc`.

### 2.3 Evaluation

```java
Evaluation eval = new Evaluation(data);
eval.crossValidateModel(classifier, data, 10, new Random(1));   // k-fold CV, classifier is cloned internally
// or
classifier.buildClassifier(train);
eval = new Evaluation(train);
eval.evaluateModel(classifier, test);                            // hold-out
```

| Method | Meaning |
|---|---|
| `pctCorrect()`, `pctIncorrect()` | Accuracy and error in percent |
| `kappa()` | Cohen's kappa |
| `precision(c)`, `recall(c)`, `fMeasure(c)` | Per class (c = class index) |
| `weightedPrecision()`, `weightedRecall()`, `weightedFMeasure()` | Weighted by class size |
| `areaUnderROC(c)`, `weightedAreaUnderROC()` | ROC AUC |
| `confusionMatrix()` | `double[][]`, rows = actual, columns = predicted |
| `toSummaryString(title, printComplexity)` | Text summary |
| `toClassDetailsString()` | Per-class TP rate, FP rate, precision, recall, F, ROC |
| `toMatrixString()` | Printable confusion matrix |
| `correlationCoefficient()` | Regression: correlation |
| `meanAbsoluteError()`, `rootMeanSquaredError()` | Regression errors |
| `relativeAbsoluteError()`, `rootRelativeSquaredError()` | Errors relative to the mean predictor, in percent |

Because `crossValidateModel` trains clones, the classifier you pass in remains untrained. Call `buildClassifier(data)` afterwards if you want a final model.

### 2.4 Saving and loading models

```java
SerializationHelper.writeAll("model.bin", new Object[] { model, new Instances(data, 0) });
Object[] parts   = SerializationHelper.readAll("model.bin");
Classifier model = (Classifier) parts[0];
Instances header = (Instances) parts[1];   // attribute structure for building new instances
```

`SerializationHelper.write(path, obj)` and `read(path)` handle a single object. Models must be loaded with the same major Weka version that wrote them.

### 2.5 Predicting a new row

Copy a row of the right structure and change values, or build one:

```java
Instance x = new DenseInstance(header.numAttributes());
x.setDataset(header);
x.setValue(header.attribute("age"), 35);
x.setValue(header.attribute("sex"), "male");   // nominal value by label
x.setClassMissing();
double idx = model.classifyInstance(x);
```

### 2.6 Options by string and by class name

```java
J48 j = new J48();
j.setOptions(Utils.splitOptions("-C 0.25 -M 2"));
Classifier c = AbstractClassifier.forName("weka.classifiers.trees.J48", Utils.splitOptions("-U"));
Classifier copy = AbstractClassifier.makeCopy(c);
```

---

## 3. Ensemble learning

Ensembles live in `weka.classifiers.meta` (plus `trees.RandomForest`).

| Ensemble | Class | Key setters (default) | Idea | Needs |
|---|---|---|---|---|
| Bagging | `meta.Bagging` | `setClassifier` (REPTree), `setNumIterations(10)`, `setBagSizePercent(100)`, `setSeed`, `setNumExecutionSlots(n)` | Train on bootstrap samples, average the votes. Reduces variance | Nominal or numeric class |
| Random forest | `trees.RandomForest` | options `-I` trees (100), `-K` features per split (0 = log2(n)+1), `-depth` (0 = unlimited), `-S` seed, `-num-slots` | Bagged random trees | Nominal or numeric class |
| AdaBoost | `meta.AdaBoostM1` | `setClassifier` (DecisionStump), `setNumIterations(10)`, `setUseResampling(false)`, `setWeightThreshold(100)` | Re-weight misclassified rows each round. Reduces bias | Nominal class |
| LogitBoost | `meta.LogitBoost` | `setClassifier` (DecisionStump), `setNumIterations(10)`, `setShrinkage(1.0)` | Additive logistic regression via boosting | Nominal class |
| Random subspace | `meta.RandomSubSpace` | `setClassifier` (REPTree), `setSubSpaceSize(0.5)`, `setNumIterations(10)` | Each model sees a random attribute subset | Nominal or numeric class |
| Stacking | `meta.Stacking` | `setClassifiers(Classifier[])`, `setMetaClassifier(Classifier)` (ZeroR), `setNumFolds(10)` | A meta model learns from out-of-fold predictions of base models | Nominal or numeric class |
| Voting | `meta.Vote` | `setClassifiers(Classifier[])`, `setCombinationRule(new SelectedTag(rule, Vote.TAGS_RULES))` | Fixed combination of base models | Majority rule needs a nominal class |

`Vote` rules: `Vote.AVERAGE_RULE` (default), `PRODUCT_RULE`, `MAJORITY_VOTING_RULE`, `MIN_RULE`, `MAX_RULE`, `MEDIAN_RULE`.

Practical notes:

- `RandomForest.setOptions(Utils.splitOptions("-I 200 -K 0 -depth 0 -S 1"))` is the most version-proof way to set the tree count, because the dedicated setter name changed between Weka generations.
- Bagging and random forests rarely overfit as trees are added; boosting can, especially on noisy data.
- Compare against the single base learner (J48 in the template) in the same cross-validation setup, with the same seed.
- For parallel training use `setNumExecutionSlots(n)` on Bagging and `-num-slots` on RandomForest.
- All ensembles are ordinary `Classifier` objects, so `Evaluation`, `SerializationHelper` and `FilteredClassifier` work with them unchanged.

---

## 4. Association rule mining

Classes are in `weka.associations`: `Apriori`, `FPGrowth`, `AssociationRules`, `AssociationRule`, `Item`.

### 4.1 Data requirements

| Algorithm | Attribute types | Class attribute |
|---|---|---|
| `Apriori` | Nominal only (discretise numeric attributes first) | None, unless mining class association rules |
| `FPGrowth` | Binary (two-valued) nominal or unary attributes. Numeric and multi-valued nominal data must be discretised and binarised first | None |

- Leave the class index unset: `data.setClassIndex(-1)`.
- Market-basket data: one column per item, nominal values `{f,t}` or `{t}` only (absent = missing). In Apriori call `setTreatZeroAsMissing(true)` when absent items are coded as the first value, so rules only mention bought items. The template builds such a dataset from a list of baskets.
- Typical preparation: `Discretize` for numeric columns, then `NominalToBinary` with `setBinaryAttributesNominal(true)` and `setTransformAllValues(true)` for FPGrowth.

### 4.2 Apriori parameters

| Setter | CLI | Default | Meaning |
|---|---|---|---|
| `setNumRules(int)` | `-N` | 10 | Number of rules to find |
| `setMetricType` / options | `-T` | 0 | Ranking metric: 0 confidence, 1 lift, 2 leverage, 3 conviction |
| `setMinMetric(double)` | `-C` | 0.9 | Minimum value of the chosen metric |
| `setDelta(double)` | `-D` | 0.05 | Step by which minimum support is lowered |
| `setUpperBoundMinSupport(double)` | `-U` | 1.0 | Starting support |
| `setLowerBoundMinSupport(double)` | `-M` | 0.1 | Lowest support to try |
| `setSignificanceLevel(double)` | `-S` | -1 | Significance test on confidence (off by default) |
| `setCar(boolean)` | `-A` | false | Class association rules only |
| `setClassIndex(int)` | `-c` | -1 | Class attribute for CAR (-1 = last, otherwise 1-based) |
| `setRemoveAllMissingCols(boolean)` | `-R` | false | Drop all-missing columns |
| `setTreatZeroAsMissing(boolean)` | `-Z` | false | Basket-style data |
| `setOutputItemSets(boolean)` | `-I` | false | Also print itemsets |

How the search works: Apriori starts at the upper support bound, finds frequent itemsets and rules, and if fewer than `numRules` rules pass the metric threshold, lowers support by `delta` and repeats, stopping at the lower bound. So the rules returned are the best `numRules` found at the highest support that produced enough.

Class association rules (`setCar(true)`) require confidence as the metric.

### 4.3 FPGrowth parameters

| Setter | CLI | Default | Meaning |
|---|---|---|---|
| `setPositiveIndex(int)` | `-P` | 2 | Which value of a binary attribute means "present" (2 = second value) |
| `setNumRulesToFind(int)` | `-N` | 10 | Number of rules |
| `setMetricType` / options | `-T` | 0 | Metric: 0 confidence, 1 lift, 2 leverage, 3 conviction |
| `setMinMetric(double)` | `-C` | 0.9 | Minimum metric value |
| `setMaxNumberOfItems(int)` | `-I` | -1 | Maximum items per itemset/rule (-1 unlimited) |
| `setUpperBoundMinSupport(double)` | `-U` | 1.0 | Starting support |
| `setLowerBoundMinSupport(double)` | `-M` | 0.1 | Lowest support |
| `setDelta(double)` | `-D` | 0.05 | Support step |
| `setFindAllRulesForSupportLevel(boolean)` | `-S` | false | Return all rules at the support level, ignoring `numRules` |
| `setRulesMustContain(String)` | `-rules-must-contain` | empty | Keep rules that include these items |

FPGrowth produces the same kind of rules as Apriori but builds a compact prefix tree (FP-tree) instead of generating candidates, which is much faster on large, dense datasets.

### 4.4 Reading results

```java
Apriori apriori = new Apriori();
apriori.buildAssociations(data);
AssociationRules rules = apriori.getAssociationRules();

for (AssociationRule r : rules.getRules()) {
    Collection<Item> premise     = r.getPremise();
    Collection<Item> consequence = r.getConsequence();
    int support                  = r.getTotalSupport();            // rows containing premise and consequence
    double primary               = r.getPrimaryMetricValue();      // metric used for ranking
    double lift                  = r.getNamedMetricValue("Lift");  // "Confidence", "Lift", "Leverage", "Conviction"
}
```

| `AssociationRule` method | Returns |
|---|---|
| `getPremise()`, `getConsequence()` | `Collection<Item>` (each `Item.toString()` looks like `outlook=sunny`) |
| `getTotalSupport()` | Count of rows matching premise and consequence |
| `getPremiseSupport()`, `getConsequenceSupport()` | Counts for each side alone |
| `getTotalTransactions()` | Dataset size |
| `getPrimaryMetricName()`, `getPrimaryMetricValue()` | Ranking metric |
| `getNamedMetricValue(String)` | Any of the four metrics (throws `Exception`) |
| `getMetricNamesForRule()` | Names of metrics available for this rule |

Metric definitions, with A the premise and B the consequence: confidence = P(B|A); lift = P(A,B) / (P(A)·P(B)), where above 1 means positive association; leverage = P(A,B) − P(A)·P(B); conviction = P(A)·P(not B) / P(A, not B).

`apriori.toString()` prints Weka's own report (itemset counts per size and the best rules), which is handy for checking your code against the Explorer.

---

## 5. Quick troubleshooting

| Message | Cause and fix |
|---|---|
| `Class index is negative (not set)!` | Call `data.setClassIndex(...)` before supervised steps |
| `weka.core.UnsupportedAttributeTypeException: Cannot handle numeric attributes!` | Apriori needs nominal data: discretise first |
| `Cannot handle numeric class!` | Classifier requires a nominal class. Use `NumericToNominal` on the class or choose a regressor |
| `Cannot handle missing class values!` | Remove rows with a missing class (`RemoveWithValues` or `data.deleteWithMissingClass()`) |
| `Training and test set are not compatible` | Test set was built or filtered with a different header. Reuse the same filter object, and load both sets with matching attributes |
| `Not enough memory` / `OutOfMemoryError` | Start the JVM with a larger heap: `java -Xmx2g -cp ...` |
| FPGrowth returns no rules | Lower `setLowerBoundMinSupport`, check that data is binary, and check `setPositiveIndex` |
| Apriori returns a handful of very long rules | Raise `setLowerBoundMinSupport` or switch the metric to lift |

### Command-line equivalents (useful for checking results)

```
java -cp weka.jar weka.classifiers.trees.J48 -t data/iris.arff -x 10
java -cp weka.jar weka.filters.unsupervised.attribute.Normalize -i data/iris.arff -o out.arff
java -cp weka.jar weka.associations.Apriori -t data/weather.nominal.arff -N 10 -C 0.9
```
