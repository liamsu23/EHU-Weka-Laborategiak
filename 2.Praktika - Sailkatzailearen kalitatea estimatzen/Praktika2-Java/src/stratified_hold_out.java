import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.core.converters.ConverterUtils.DataSink;

import java.io.File;


public class stratified_hold_out {
    public static void main(String[] args) throws Exception {
        if (args.length < 3 ){
            System.out.println("Por favor, especifica la ruta al archivo .arff y la ruta de salida.");
            return;
        }
        try {
            // Cargar el archivo .arff
            String filePath = args[0];
            DataSource source = new DataSource(filePath);
            Instances data = source.getDataSet();

            // Definir la clase objetivo
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // 1. Aplicar Stratified Hold-Out (80% entrenamiento, 20% prueba)

            // === CREAR CONJUNTO DE ENTRENAMIENTO (80%) ===
            StratifiedRemoveFolds stratifiedTrainFilter = new StratifiedRemoveFolds();
            stratifiedTrainFilter.setNumFolds(5);  // Dividir en 5 partes (cada una es 20%)
            stratifiedTrainFilter.setFold(1);  // Elegimos 1 fold como prueba (20%)
            stratifiedTrainFilter.setInvertSelection(true);  // Nos quedamos con el 80% (4 folds)
            stratifiedTrainFilter.setInputFormat(data);
            Instances trainData = Filter.useFilter(data, stratifiedTrainFilter);

            // === CREAR CONJUNTO DE PRUEBA (20%) ===
            StratifiedRemoveFolds stratifiedTestFilter = new StratifiedRemoveFolds();
            stratifiedTestFilter.setNumFolds(5);  // Mismo número de folds
            stratifiedTestFilter.setFold(1);  // Elegimos el mismo fold como prueba
            stratifiedTestFilter.setInvertSelection(false);  // Nos quedamos con el 20% (1 fold)
            stratifiedTestFilter.setInputFormat(data);
            Instances testData = Filter.useFilter(data, stratifiedTestFilter);

            // Guardar conjuntos en archivos .arff
            String trainFilePath = args[1]; // Ruta para el conjunto de entrenamiento
            String testFilePath = args[2];  // Ruta para el conjunto de prueba

            // Guardar conjuntos en archivos .arff
            saveInstancesToFile(trainData, args[1]);  // Guardar entrenamiento
            saveInstancesToFile(testData, args[2]);   // Guardar prueba

            System.out.println("Archivos guardados:");
            System.out.println("Entrenamiento: " + trainFilePath);
            System.out.println("Prueba: " + testFilePath);

        }
        catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error al cargar el archivo .arff");
        }
    }

    // Método para guardar Instances en un archivo .arff
    private static void saveInstancesToFile(Instances data, String filePath) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);

        // Crear el directorio si no existe
        File outputFile = new File(filePath);
        outputFile.getParentFile().mkdirs();

        saver.setFile(outputFile);
        saver.writeBatch();
    }
}
