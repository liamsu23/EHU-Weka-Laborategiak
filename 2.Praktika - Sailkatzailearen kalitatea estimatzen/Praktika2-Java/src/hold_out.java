import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;


import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Date;

public class hold_out {
    // Aurrebaldintzak:
    //      1. argumentuan .arff fitxategi baten path-a hartzen da. Fitxategi horren klasea azken atributuan dator.
    //      2. argumentuan irteerarako emaitzak gordetzeko fitxategi baten path-a ematen da
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("Por favor, especifica la ruta al archivo .arff y la ruta de salida.");
            return;
        }
        try {
            // Argumentuetatik .arff artxiboa kargatu
            String filePath = args[0];
            DataSource source = new DataSource(filePath);
            Instances data = source.getDataSet();

            // Klasea ezarri
            if (data.classIndex()==-1) {
                data.setClassIndex(data.numAttributes()-1);
            }

            // 1.Urratsa: Datuak randomizatu
                // Crear y configurar el filtro de randomización
            Randomize randomizeFilter = new Randomize();
            randomizeFilter.setRandomSeed(42);
            randomizeFilter.setInputFormat(data);
                // Aplicar el filtro a los datos
            Instances randomizedData = Filter.useFilter(data, randomizeFilter);


            // 2.Urratsa: Datuak zatitu
                // Crear el filtro RemovePercentage para dividir los datos
            RemovePercentage removeTrain = new RemovePercentage();
            removeTrain.setPercentage(34); // Eliminar 34% → 66% queda para entrenamiento
            removeTrain.setInputFormat(randomizedData);
            Instances trainData = Filter.useFilter(randomizedData, removeTrain);

            RemovePercentage removeTest = new RemovePercentage();
            removeTest.setPercentage(66); // Eliminar 66% → 34% queda para prueba
            removeTest.setInvertSelection(true); // Invertir selección → quedarnos con el 34%
            removeTest.setInputFormat(randomizedData);
            Instances testData = Filter.useFilter(randomizedData, removeTest);

            // 3.Urratsa: Sailkatzailea sortu eta entrenatu
            Classifier classifier = new NaiveBayes();
            classifier.buildClassifier(trainData);

            // 4.Urratsa: Modeloa ebaluatu
            Evaluation eval = new Evaluation(trainData);
            eval.evaluateModel(classifier, testData);

            // 5.Urratsa: Emaitzak gorde
            String outputPath = args[1] + "/emaitzak_hold-out.txt"; // Ruta del segundo argumento + nombre del archivo
            try (BufferedWriter buffer = new BufferedWriter(new FileWriter(outputPath))) {
                // Exekuzio data
                Date date = new Date();
                buffer.write("Exekuzio data: " + date.toString() + "\n");

                // Exukuziorako argumentuak
                buffer.write("\nExekuzio argumentuak: \n");
                buffer.write("1. " + args[0] + "\n" + "2. " + args[1] + "\n");

                // Klase minoritarioari dagokion Precision, Recall, F-score
                    // Identificar la clase minoritaria
                int classIndex = data.classIndex();
                int[] classCounts = data.attributeStats(classIndex).nominalCounts;
                int minClassIndex = -1;
                int minInstances = Integer.MAX_VALUE;
                for (int i = 0; i < classCounts.length; i++) {
                    if (classCounts[i] < minInstances && classCounts[i] > 0) {
                        minInstances = classCounts[i];
                        minClassIndex = i;
                    }
                }
                if (minClassIndex != -1) {
                    String minorClass = data.classAttribute().value(minClassIndex);
                    buffer.write("\nKlase Minoritarioa: (" + minorClass + ")\n");
                    buffer.write(String.format("Precision: %.3f\n", eval.precision(minClassIndex)));
                    buffer.write(String.format("Recall: %.3f\n", eval.recall(minClassIndex)));
                    buffer.write(String.format("F-Score: %.3f\n", eval.fMeasure(minClassIndex)));
                }

                // Weighted Avg Precision, Recall y F-score
                buffer.write("\n=== Weighted Average Metrics ===\n");
                buffer.write(String.format("Weighted Precision: %.3f\n" , eval.weightedPrecision()));
                buffer.write(String.format("Weighted Recall: %.3f\n" , eval.weightedRecall()));
                buffer.write(String.format("Weighted F-Score: %.3f\n" , eval.weightedFMeasure()));

                // Conffusion Matrix
                double[][] confusionMatrix = eval.confusionMatrix();
                String[] classNames = new String[data.numClasses()];
                for (int i = 0; i < data.numClasses(); i++) {
                    classNames[i] = data.classAttribute().value(i);
                }
                buffer.write("\n=== Confusion Matrix ===\n");
                buffer.write(eval.toMatrixString());

            }
            catch (IOException e) {
                e.printStackTrace();
                System.out.println("Error al guardar los resultados en el archivo.");
            }

        }
        catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error al cargar el archivo .arff");
        }
    }
}
