import weka.core.AttributeStats;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
public class Praktika1 {
    // Aurrebaldintzak: 1. argumentuan .arff fitxategi baten path-a hartzen da.
    //                  Fitxategi horren klasea azken atributuan dator.
    public static void main(String[] args) throws Exception {
        if (args.length < 1) {
            System.out.println("Por favor, especifica la ruta al archivo .arff como argumento.");
            return;
        }
        // -Argumentuetatik .arff artxiboa kargatu
        String filePath = args[0];
        try {
            // 1. Datuak kargatu
            DataSource source = new DataSource(filePath);
            Instances data = source.getDataSet();

            // 2. Klasea ezarri
            if (data.classIndex() == -1) {
                data.setClassIndex(data.numAttributes() - 1);
            }
            System.out.println("Aukeratutako klasea: " + data.classAttribute().name());

            // -Aztertzen ari garen fitxategiko path-a
            System.out.println("Path del archivo: " + filePath);

            // -Instantzia eta atributu kopurua
            System.out.println("Instantzia kopurua: " + data.numInstances());
            System.out.println("Atributu kopurua: " + data.numAttributes());

            // -Lehenengo atributuak har ditzakeen balio ezberdinak
            System.out.println("Lehenengo atributuak har ditzakeen balioak:");
            int firstAttributeIndex = 0;
            AttributeStats firstAttributeStats = data.attributeStats(firstAttributeIndex);
            System.out.println("\tBalio ezberdin kopurua: " + firstAttributeStats.distinctCount);

            // -Azken atributuak hartzen dituen balioak eta maiztasuna
            // Obtener estadísticas del atributo de clase
            int classAttributeIndex = data.classIndex();
            AttributeStats classAttributeStats = data.attributeStats(classAttributeIndex);
            // Verificar si es nominal
            if (classAttributeStats.nominalCounts == null) {
                System.out.println("El atributo de clase no es nominal. Este cálculo no es aplicable.");
                return;
            }
            // Obtener los valores nominales y sus frecuencias
            int[] classCounts = classAttributeStats.nominalCounts;
            String minorClass = null;
            int minorCount = Integer.MAX_VALUE;

            System.out.println("\nValores nominales del atributo de clase y sus frecuencias:");
            for (int i = 0; i < classCounts.length; i++) {
                String classValue = data.classAttribute().value(i);
                int count = classCounts[i];
                System.out.println("\t" + classValue + ": " + count + " instancias");
                // Identificar la clase minoritaria basada en la frecuencia
                if (count < minorCount && count > 0) {
                    minorCount = count;
                    minorClass = classValue;
                }
            }
            // Mostrar la clase minoritaria
            System.out.println("\nClase minoritaria: " + minorClass + " (" + minorCount + " instancias)");

            // -Azken aurreko atributuak dituen missing value kopurua
            System.out.println("\nAzken aurreko atributuak dituen missing value kopurua:");
            int secondLastAttributeIndex = data.numAttributes() - 2;
            int missingCount = data.attributeStats(secondLastAttributeIndex).missingCount;
            System.out.println("\tMissing values: " + missingCount);

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error al cargar el archivo .arff.");
        }
    }
}
