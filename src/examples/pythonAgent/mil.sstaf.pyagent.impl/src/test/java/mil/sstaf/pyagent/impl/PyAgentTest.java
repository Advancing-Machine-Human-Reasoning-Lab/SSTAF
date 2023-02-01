package mil.sstaf.pyagent.impl;

import mil.sstaf.core.features.FeatureConfiguration;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;


public class PyAgentTest {

        @Test
        @DisplayName("Confirm that PyAgent accepts and returns JSON string")
        void test1() {
            String prompts = "{'animal': ['dog', 'cat'], 'fruit': ['apple', 'orange'], 'animal': ['rabbit', 'parrot']}";
            FeatureConfiguration fc = FeatureConfiguration.builder().build();

            PyAgentImpl predictWord = new PyAgentImpl();
            predictWord.configure(fc);
            String results = predictWord.predictWord(prompts);
            System.out.println(results);
            Assertions.assertEquals(prompts.isBlank(), results.isBlank());
        }
}
