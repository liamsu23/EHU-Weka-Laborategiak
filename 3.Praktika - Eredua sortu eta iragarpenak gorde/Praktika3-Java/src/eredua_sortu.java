import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;

public class eredua_sortu {

    public static void main(String[] args){
        if (args.length < 3){
            System.out.println("Erabilera: java -jar EreduaSortu.jar data.arff NB.model KalitatearenEstimazioa.txt");
        }

        String dataPath = args[0];  // Archivo de datos
        String modelPath = args[1];  // Ruta donde se guardará el modelo
        String outputPath = args[2];  // Archivo donde se guardarán los resultados

        try {
            // Datuak kargatu
            DataSource source = new DataSource(dataPath);
            Instances data = source.getDataSet();

            // Klasea ezarri
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }

            // Crear el modelo Naive Bayes
            Classifier classifier = new NaiveBayes();
            classifier.buildClassifier(data);

            // Guardar el modelo entrenado
            SerializationHelper.write(modelPath, classifier);
            System.out.println("Modelo guardado en: " + modelPath);

            //TODO

        }
        catch(Exception e){
            e.printStackTrace();
            System.out.println("Errorea prozesuan");
        }
    }
}
