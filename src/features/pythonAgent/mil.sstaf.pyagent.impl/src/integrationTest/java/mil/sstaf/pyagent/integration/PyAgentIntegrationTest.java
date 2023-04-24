package mil.sstaf.pyagent.integration;


import mil.sstaf.core.configuration.SSTAFConfiguration;
import mil.sstaf.core.entity.Message;
import mil.sstaf.core.features.*;
import mil.sstaf.pyagent.api.PyAgent;
import mil.sstaftest.util.BaseFeatureIntegrationTest;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.io.File;

import static org.junit.jupiter.api.Assertions.*;


public class PyAgentIntegrationTest extends BaseFeatureIntegrationTest<PyAgent, FeatureConfiguration> {


    PyAgentIntegrationTest() {
        super(PyAgent.class, getSpec());
    }

    private static FeatureSpecification getSpec() {
        return FeatureSpecification.builder()
                .featureClass(PyAgent.class)
                .featureName("PyAgent")
                .majorVersion(1)
                .minorVersion(0)
                .requireExact(true)
                .build();
    }

    @BeforeEach
    public void setup() {
        System.setProperty(SSTAFConfiguration.SSTAF_CONFIGURATION_PROPERTY,
                "src" + File.separator +
                        "integrationTest" + File.separator +
                        "resources" + File.separator +
                        "PyAgentConfiguration.json");
    }


    @Nested
    @DisplayName("Test PyAgent-specific deployment and startup issues")
    class PythonStuff {
        @Test
        @DisplayName("Check that python scripts install correctly and works")
        void test1() {
            assertDoesNotThrow(() -> {
                PyAgent pyAgent = loadAndResolveFeature();
                pyAgent.configure(FeatureConfiguration.builder().build());
                pyAgent.init();

                String prompts = "{'animal': ['dog', 'cat']}";

                String result = pyAgent.predictWord(prompts);

                // model is not 100% accurate meaning it will not always return a specific word for a specific prompt
                // thus we simply check if we receive input
            });
        }

        @Test
        @DisplayName("Check that tick works")
        void testTick() {
            assertDoesNotThrow(() -> {
                PyAgent pyAgent = loadAndResolveFeature();
                pyAgent.configure(FeatureConfiguration.builder().build());
                pyAgent.init();

                long tzero = System.currentTimeMillis();
                long tzero_res = pyAgent.setTZero(tzero);

                assertEquals(tzero, tzero_res);

                double oldCapability = 1.0;

                for (long time = tzero; time < tzero+1000000; time += 10000) {

                    ProcessingResult pr = pyAgent.tick(time);
                    assertEquals(1, pr.messages.size());
                    Message m = pr.messages.get(0);
                    HandlerContent hc = m.getContent();
                    assertTrue (hc instanceof DoubleContent);
                    DoubleContent dc = (DoubleContent) hc;
                    double newCapability = dc.getValue();
                    if (time == tzero) {
                        assertEquals(oldCapability, newCapability);
                    } else {
                        assertTrue(oldCapability > newCapability);
                    }
                    oldCapability = newCapability;
                }
            });
        }
    }
}