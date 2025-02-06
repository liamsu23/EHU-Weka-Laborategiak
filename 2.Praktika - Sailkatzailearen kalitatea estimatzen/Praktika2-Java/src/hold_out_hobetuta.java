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

public class hold_out_hobetuta {
    // Aurrebaldintzak:
    //      1. argumentuan .arff fitxategi baten path-a hartzen da. Fitxategi horren klasea azken atributuan dator.
    //      2. argumentuan irteerarako emaitzak gordetzeko fitxategi baten path-a ematen da
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("Por favor, especifica la ruta al archivo .arff y la ruta de salida.");
            return;
        }

        try {
            // Cargar el archivo .arff
            String filePath = args[0];
            DataSource source = new DataSource(filePath);
            Instances data = source.getDataSet();

            // Establecer la clase
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // 1. Aleatorizar los datos
            Randomize randomizeFilter = new Randomize();
            randomizeFilter.setRandomSeed(42);
            randomizeFilter.setInputFormat(data);
            Instances randomizedData = Filter.useFilter(data, randomizeFilter);

            // 2. Dividir los datos en entrenamiento y prueba
            Instances trainData = splitData(randomizedData, 34); // 34% para prueba, 66% para entrenamiento
            Instances testData = splitData(randomizedData, 66);  // 66% para prueba, 34% para entrenamiento

            // 3. Crear y entrenar el clasificador
            Classifier classifier = new NaiveBayes();
            classifier.buildClassifier(trainData);

            // 4. Evaluar el modelo
            Evaluation eval = new Evaluation(trainData);
            eval.evaluateModel(classifier, testData);

            // 5. Guardar los resultados
            String outputPath = args[1] + "/emaitzak_hold-out-hobetuta.txt"; // Ruta del archivo de salida
            try (BufferedWriter buffer = new BufferedWriter(new FileWriter(outputPath))) {
                // Escribir la fecha de ejecución y los parámetros de entrada
                Date date = new Date();
                buffer.write("Exekuzio data: " + date.toString() + "\n");
                buffer.write("\nExekuzio argumentuak: \n");
                buffer.write("1. " + args[0] + "\n" + "2. " + args[1] + "\n");

                // Escribir las métricas y la matriz de confusión
                writeMetrics(buffer, eval, data);
            } catch (IOException e) {
                e.printStackTrace();
                System.out.println("Error al guardar los resultados en el archivo.");
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error al cargar el archivo .arff");
        }
    }

    // Método para dividir los datos en entrenamiento y prueba según el porcentaje
    private static Instances splitData(Instances data, double percentage) throws Exception {
        RemovePercentage filter = new RemovePercentage();
        filter.setPercentage(percentage);
        filter.setInputFormat(data);
        return Filter.useFilter(data, filter);
    }

    // Método para escribir las métricas en el archivo
    private static void writeMetrics(BufferedWriter buffer, Evaluation eval, Instances data) throws IOException {
        // Obtener la clase minoritaria
        int minClassIndex = getMinorClassIndex(data);
        if (minClassIndex != -1) {
            String minorClass = data.classAttribute().value(minClassIndex);
            buffer.write("\nKlase Minoritarioa: (" + minorClass + ")\n");
            buffer.write(String.format("Precision: %.3f\n", eval.precision(minClassIndex)));
            buffer.write(String.format("Recall: %.3f\n", eval.recall(minClassIndex)));
            buffer.write(String.format("F-Score: %.3f\n", eval.fMeasure(minClassIndex)));
        }

        // Métricas de precisión, recall y F-score ponderadas
        buffer.write("\n=== Weighted Average Metrics ===\n");
        buffer.write(String.format("Weighted Precision: %.3f\n", eval.weightedPrecision()));
        buffer.write(String.format("Weighted Recall: %.3f\n", eval.weightedRecall()));
        buffer.write(String.format("Weighted F-Score: %.3f\n", eval.weightedFMeasure()));

        // Matriz de confusión
        double[][] confusionMatrix = eval.confusionMatrix();
        String[] classNames = new String[data.numClasses()];
        for (int i = 0; i < data.numClasses(); i++) {
            classNames[i] = data.classAttribute().value(i);
        }
        buffer.write("\n=== Confusion Matrix ===\n");
        for (String className : classNames) {
            buffer.write(className + " ");
        }
        buffer.write("<-- classified as\n");
        for (int i = 0; i < confusionMatrix.length; i++) {
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                buffer.write(" " + confusionMatrix[i][j]);
            }
            buffer.write(" |     " + classNames[i] + "\n");
        }
    }


    // Método para obtener el índice de la clase minoritaria
    private static int getMinorClassIndex(Instances data) {
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
        return minClassIndex;
    }

}
