import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.Random;

public class ParametroEkorketa {
        public static void main(String[] args) throws Exception {

        if (args.length<1){
            System.out.println("Mesedez dauten path-a sartu.");
            return;
        }

        String inputPath=args[0];

        //1. Datuak kargatu
        DataSource source = new DataSource(inputPath);
        Instances data = source.getDataSet();

        //2. Klasea ezarri
        if (data.classIndex()==-1){
            data.setClassIndex(data.numAttributes()-1);
        }

        //3. Sailkatzailea definitu
        J48 classifier = new J48();

        //4. Configurar búsqueda de parámetros con validación cruzada
        CVParameterSelection paramSearch = new CVParameterSelection();
        paramSearch.setClassifier(classifier);

        //5. Especificar los parámetros a optimizar y su rango de valores
        paramSearch.addCVParameter("C 0.1 0.5 5");  // Ajustamos el rango de confianza
        paramSearch.addCVParameter("M 1 5 5");  // Evitamos valores extremos

        //6. Evaluar con 3-fold Cross-Validation
        Evaluation eval = new Evaluation(data);
        paramSearch.buildClassifier(data);
        eval.crossValidateModel(paramSearch, data, 3, new Random(1));

        //7. Imprimir resultados
        System.out.println("Mejores parámetros encontrados:");
        //System.out.println(java.util.Arrays.toString(paramSearch.getBestClassifierOptions()));
        System.out.println(String.join(" ", paramSearch.getBestClassifierOptions()));
        System.out.println("Precisión: " + eval.pctCorrect());


    }
}
