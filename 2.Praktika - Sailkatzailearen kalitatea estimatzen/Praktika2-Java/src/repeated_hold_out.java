import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Date;

public class repeated_hold_out {
    // Número de repeticiones
    private static final int NUM_REPETITIONS = 10;

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("Por favor, especifica la ruta al archivo .arff y la ruta de salida.");
            return;
        }

        try {
            // Argumentuetatik .arff artxiboa kargatu
            String filePath = args[0];
            ConverterUtils.DataSource source = new ConverterUtils.DataSource(filePath);
            Instances data = source.getDataSet();

            // Klasea ezarri
            if (data.classIndex()==-1) {
                data.setClassIndex(data.numAttributes()-1);
            }

            // Variables para acumular métricas
            double totalPrecision = 0, totalRecall = 0, totalFScore = 0;
            double totalWeightedPrecision = 0, totalWeightedRecall = 0, totalWeightedFScore = 0;

            for (int i = 0; i < NUM_REPETITIONS; i++) {
                System.out.println(i);
                // 1.Urratsa: Datuak randomizatu
                // Crear y configurar el filtro de randomización
                Randomize randomizeFilter = new Randomize();
                randomizeFilter.setRandomSeed(1);
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

                // 4.Urratsa: Sailkatzailea probatu
                Evaluation evaluation = new Evaluation(trainData);
                evaluation.evaluateModel(classifier, testData);

                // 5.Urratsa: Emaitzak gorde
                // Identificar la clase minoritaria
                int classIndex = data.classIndex();
                int[] classCounts = data.attributeStats(classIndex).nominalCounts;
                int minClassIndex = -1;
                int minInstances = Integer.MAX_VALUE;
                for (int j = 0; j < classCounts.length; j++) {
                    if (classCounts[j] < minInstances && classCounts[j] > 0) {
                        minInstances = classCounts[j];
                        minClassIndex = j;
                    }
                }
                if (minClassIndex != -1) {
                    totalPrecision += evaluation.precision(minClassIndex);
                    totalRecall += evaluation.recall(minClassIndex);
                    totalFScore += evaluation.fMeasure(minClassIndex);
                }

                totalWeightedPrecision += evaluation.weightedPrecision();
                totalWeightedRecall += evaluation.weightedRecall();
                totalWeightedFScore += evaluation.weightedFMeasure();
            }

            // Calcular los promedios de las métricas
            double avgPrecision = totalPrecision / NUM_REPETITIONS;
            double avgRecall = totalRecall / NUM_REPETITIONS;
            double avgFScore = totalFScore / NUM_REPETITIONS;
            double avgWeightedPrecision = totalWeightedPrecision / NUM_REPETITIONS;
            double avgWeightedRecall = totalWeightedRecall / NUM_REPETITIONS;
            double avgWeightedFScore = totalWeightedFScore / NUM_REPETITIONS;

            // Guardar los resultados
            String outputPath = args[1] + "/emaitzak_repeated_hold-out.txt";
            try (BufferedWriter buffer = new BufferedWriter(new FileWriter(outputPath))) {
                Date date = new Date();
                buffer.write("Exekuzio data: " + date.toString() + "\n");
                buffer.write("\nExekuzio argumentuak: \n");
                buffer.write("1. " + args[0] + "\n" + "2. " + args[1] + "\n");
                buffer.write("\n=== Average Metrics over " + NUM_REPETITIONS + " runs ===\n");
                buffer.write(String.format("Avg Precision (Minor Class): %.3f\n", avgPrecision));
                buffer.write(String.format("Avg Recall (Minor Class): %.3f\n", avgRecall));
                buffer.write(String.format("Avg F-Score (Minor Class): %.3f\n", avgFScore));
                buffer.write("\n=== Weighted Average Metrics ===\n");
                buffer.write(String.format("Avg Weighted Precision: %.3f\n", avgWeightedPrecision));
                buffer.write(String.format("Avg Weighted Recall: %.3f\n", avgWeightedRecall));
                buffer.write(String.format("Avg Weighted F-Score: %.3f\n", avgWeightedFScore));
            }

            System.out.println("Evaluación completada y resultados guardados en: " + outputPath);
        }
        catch (Exception e){
            e.printStackTrace();
            System.out.println("Error: " + e.getMessage());
        }
    }
}
