import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Date;

public class k_fCV {
    // Aurrebaldintzak: 1. argumentuan .arff fitxategi baten path-a hartzen da. Fitxategi horren klasea azken atributuan dator.
    //                  2. argumentuan irteerarako emaitzak gordetzeko fitxategi baten path-a ematen da
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("Por favor, especifica la ruta al archivo .arff como argumento.");
            return;
        }

        try {
            // Argumentuetatik .arff artxiboa kargatu
            String filePath = args[0];
            DataSource source = new DataSource(filePath);
            Instances data = source.getDataSet();
            // Klasea ezarri
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // Sailkatzailea sortu eta entrenatu
            Classifier classifier = new NaiveBayes();
            classifier.buildClassifier(data);

            // Modeloa ebaluatu k-fold-crossvalidation bidez
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(classifier,data,5,new java.util.Random(1));


            // Guardar resultados en un archivo
            String outputPath = args[1] + "/emaitzak_k-fCV.txt"; // Ruta del segundo argumento + nombre del archivo
            try (BufferedWriter buffer = new BufferedWriter(new FileWriter(outputPath))) {

                buffer.write(eval.toMatrixString());

                // -Precision metrika klasearen balio bakoitzeko eta weighted avg
                // Klase bakoitzeko precision idatzi
                buffer.write("\n=== Precision Klase bakoitzkeo eta Weighted Avg ===\n");
                for (int i = 0; i < data.numClasses(); i++) {
                    String classValue = data.classAttribute().value(i);
                    double precision = eval.precision(i);
                    buffer.write("Klasea: " + classValue + " -> Precision: " + String.format("%.3f", precision) + "\n");
                }
                // Weighted avg precision idatzi
                buffer.write(String.format("\nWeighted Avg Precision: %.3f\n", eval.weightedPrecision()));
                //buffer.write("W Avg Precision: " + eval.weightedPrecision());

                // -Exekuzio data
                buffer.write("\n=== Exekuzio data ===\n");
                Date gaur = new Date();
                buffer.write(gaur.toString() + "\n");

                // -Exekuziorako argumentuak
                buffer.write("\n=== Exekuziorako argumentuak ===\n");
                buffer.write("1. " + args[0] + "\n");
                buffer.write("2. " + args[1] + "\n");

                // -Ebaluazio emaitzak
                buffer.write(eval.toSummaryString("\n=== Ebaluazio emaitzak ===\n" ,false));

            } catch (IOException e) {
                e.printStackTrace();
                System.out.println("Error al guardar el archivo en la ruta especificada.");
            }


        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error al cargar el archivo .arff.");
        }
    }
}
