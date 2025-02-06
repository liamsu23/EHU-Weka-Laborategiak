import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Date;

public class stratified_hold_out_2atala {
    public static void main(String[] args) {
        if (args.length < 3 ){
            System.out.println("Por favor, especifica la ruta al archivo .arff y la ruta de salida.");
            return;
        }

        String trainPath = args[0];
        String devPath = args[1];
        String outputPath = args[2];

        try {
            // Cargar conjunto de entrenamiento
            DataSource trainSource = new DataSource(trainPath);
            Instances trainData = trainSource.getDataSet();
            // Definir la clase objetivo (último atributo)
            if (trainData.classIndex() == -1) {
                trainData.setClassIndex(trainData.numAttributes() - 1);
            }

            // Cargar conjunto de validación
            DataSource devSource = new DataSource(devPath);
            Instances devData = devSource.getDataSet();
            // Asegurar que devData tiene la misma estructura
            if (devData.classIndex() == -1) {
                devData.setClassIndex(devData.numAttributes() - 1);
            }

            // Entrenar modelo Naive Bayes
            Classifier classifier = new NaiveBayes();
            classifier.buildClassifier(trainData);

            // Evaluar el modelo
            Evaluation eval = new Evaluation(trainData);
            eval.evaluateModel(classifier, devData);

            // Guardar resultados en evaluation.txt
            saveEvaluationResults(outputPath, trainPath, devPath, eval);

            System.out.println("Evaluación completada. Resultados guardados en: " + outputPath);

        }
        catch (Exception e){
            e.printStackTrace();
            System.out.println("Error al cargar el archivo .arff");
        }

    }

    // Método para guardar la evaluación en un archivo de texto
    private static void saveEvaluationResults(String outputPath, String trainPath, String devPath, Evaluation eval) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(outputPath))) {
            // Obtener la fecha y hora actual
            String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());

            // Escribir en el archivo
            writer.println("Fecha de ejecución: " + timestamp);
            writer.println("Archivos usados:");
            writer.println("  - Train: " + trainPath);
            writer.println("  - Dev: " + devPath);
            writer.println("\nMatriz de confusión:");
            writer.println(eval.toMatrixString());
            writer.println("Accuracy: " + String.format("%.2f%%", eval.pctCorrect()));

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error al escribir en el archivo de salida.");
        }
    }
}
