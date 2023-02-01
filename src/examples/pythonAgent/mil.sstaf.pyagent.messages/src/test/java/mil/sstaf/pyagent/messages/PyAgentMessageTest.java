package mil.sstaf.pyagent.messages;

import org.junit.jupiter.api.*;

public class PyAgentMessageTest {

    @Nested
    @DisplayName("Happy Path")
    class HappyPath {
        @Test
        @DisplayName("Confirm that a PredictWordRequest can be made")
        void test1() {
            String prompts = "{'animal': ['dog', 'cat'], 'fruit': ['apple', 'orange'], 'animal': ['rabbit', 'parrot']}";

            PredictWordRequest pwr = PredictWordRequest.builder().prompts(prompts).build();

            Assertions.assertEquals(prompts, pwr.prompts);
        }
    }
}
