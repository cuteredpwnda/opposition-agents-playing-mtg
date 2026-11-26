import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.semanticweb.HermiT.Reasoner;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.profiles.OWL2DLProfile;
import org.semanticweb.owlapi.profiles.OWLProfileReport;
import org.semanticweb.owlapi.reasoner.OWLReasoner;

public class ReasonerCheck {
    private static String json(String value) {
        return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"")
            .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") + "\"";
    }

    public static void main(String[] args) throws Exception {
        OWLOntology ontology = OWLManager.createOWLOntologyManager()
            .loadOntologyFromOntologyDocument(new File(args[0]));
        OWLProfileReport profile = new OWL2DLProfile().checkOntology(ontology);
        List<String> violations = new ArrayList<>();
        profile.getViolations().forEach(v -> violations.add(json(v.toString())));
        Collections.sort(violations);
        System.out.println("PROFILE={\"in_profile\":" + profile.isInProfile()
            + ",\"violations\":[" + String.join(",", violations) + "]}");
        OWLReasoner reasoner = new Reasoner.ReasonerFactory().createReasoner(ontology);
        try {
            boolean consistent = reasoner.isConsistent();
            System.out.println("CONSISTENCY=" + consistent);
            List<String> unsatisfiable = new ArrayList<>();
            if (consistent) {
                reasoner.getUnsatisfiableClasses().getEntitiesMinusBottom()
                    .forEach(c -> unsatisfiable.add(json(c.getIRI().toString())));
            }
            Collections.sort(unsatisfiable);
            System.out.println("REASONER={\"consistent\":" + consistent
                + ",\"unsatisfiable_classes\":[" + String.join(",", unsatisfiable) + "]}");
        } finally {
            reasoner.dispose();
        }
    }
}
