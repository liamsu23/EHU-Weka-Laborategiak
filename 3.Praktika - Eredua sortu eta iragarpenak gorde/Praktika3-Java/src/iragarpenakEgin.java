import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.BufferedWriter;
import java.io.FileWriter;

public class iragarpenakEgin {
    public static void main(String[] args){
        if (args.length < 3){
            System.out.println("Argumentuak ondo sartu.");
            return;
        }
        String modelPath = args[0];        // Ruta del modelo Naive Bayes
        String testFilePath = args[1];     // Ruta del archivo test_blind.arff
        String outputPath = args[2];       // Ruta del archivo de predicciones

        try {
            // 1 Cargar el modelo Naive Bayes entrenado
            Classifier model = (Classifier) weka.core.SerializationHelper.read(modelPath);
            System.out.println("Modelo cargado correctamente desde: " + modelPath);

            // 2 Cargar el conjunto de test
            DataSource source = new DataSource(testFilePath);
            Instances testInstances = source.getDataSet();
            if (testInstances.classIndex() == -1) {
                testInstances.setClassIndex(testInstances.numAttributes() - 1);
            }

            // 3 Abrir el archivo para escribir las predicciones
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {
                writer.write("Instancia,Predicción\n");

                // 4 Realizar predicciones sobre cada instancia
                for (int i = 0; i < testInstances.numInstances(); i++) {
                    double predIndex = model.classifyInstance(testInstances.instance(i));
                    String predictedClass = testInstances.classAttribute().value((int) predIndex);

                    // Escribir la predicción en el archivo
                    writer.write((i + 1) + "," + predictedClass + "\n");
                }

                System.out.println("Predicciones guardadas en: " + outputPath);
            }

        }
        catch (Exception e){
            e.printStackTrace();
            System.out.println("Errorea.");
        }
    }
}
