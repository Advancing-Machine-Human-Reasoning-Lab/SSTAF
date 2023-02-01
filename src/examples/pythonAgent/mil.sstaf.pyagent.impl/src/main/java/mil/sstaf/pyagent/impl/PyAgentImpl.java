/**
 * Template implementation of an Agent.
 */
package mil.sstaf.pyagent.impl;

import mil.sstaf.core.entity.Address;
import mil.sstaf.core.features.*;
import mil.sstaf.core.util.SSTAFException;
import mil.sstaf.pyagent.api.PyAgent;
import mil.sstaf.pyagent.messages.PredictWordRequest;
import mil.sstaf.pyagent.messages.PredictWordResult;
import org.slf4j.LoggerFactory;
import org.slf4j.Logger;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PyAgentImpl extends BaseAgent implements PyAgent {
    private static final Logger logger = LoggerFactory.getLogger(PyAgentImpl.class);
    AppAdapter pythonService;

    private final AtomicInteger requestCount = new AtomicInteger(0);

    /**
     * Features require a no-args constructor to be loaded by ServiceLoader.
     */
    public PyAgentImpl() {
        super(/* Feature Name */ "PyAgent",
                /* Major Version */1,
                /* Minor Version */0,
                /* Patch Version */0,
                /* Requires Configuration? */ false,
                /* Description */ "This is an example of a Python-based model");
    }


    @Override
    public void configure(FeatureConfiguration configuration) {
        super.configure(configuration);
        final String resourcePath = getClass().getPackage().getName().replace('.', '/') + "/";

        /*
         * Configure support for the helper application.
         * Every required application, script and data file must be
         * declared to be extracted. Helper applications can't read
         * resources that are packaged in the jar file.
         */
        AppConfiguration config = AppConfiguration.builder()
                //
                // Applications with their own REPLs should be declared
                // as DURABLE. One-and-done applications, even if long-running,
                // should be declared as TRANSIENT
                //
                .mode(AppSupport.Mode.DURABLE)
                .resource(resourcePath + "sstaf_pyagent.py")
                .resourceOwner(this.getClass())
                //
                // Specify the helper application and its arguments
                //
                .processArgs(List.of("python", "sstaf_pyagent.py"))
                .build();
        pythonService = AppSupport.createAdapter(config);
        ResourceManager rm = pythonService.getResourceManager();
        logger.info("Script installation directory = {}", rm.getDirectory());
        for (var entry : rm.getResourceFiles().entrySet()) {
            logger.info(entry.getKey() + " -->" + entry.getValue().getPath());
        }
    }

    /**
     * Tell the framework which messages this Feature accepts
     * @return the list of messages
     */
    @Override
    public List<Class<? extends HandlerContent>> contentHandled() {
        return List.of(PredictWordRequest.class);
    }

    /**
     * Generates common "front matter" for commands
     * @param command the command to execute
     * @return a partially-loaded StringBuilder
     */
    protected StringBuilder generatePrefix(String command) {
        StringBuilder sb = new StringBuilder();
        sb.append(requestCount.getAndIncrement()).append(' ');
        sb.append(command).append(' ');
        return sb;
    }

    /**
     * Example implementation for dispatching a request to the Python script
     * <p>
     * Invoking the script requires:
     * <ol>
     * <li>
     *     Generating the command as a String;
     * </li>
     * <li>
     *     Invoking the script using the AppSession object
     * </li>
     * <li>
     *     Parsing the result String
     * </li>
     * </ol>
     *
     * This method dispatches a list of Strings to the Python app. The
     * application returns the sum of the length of each arg.
     * @param prompts the Strings to count
     * @return predicted words in a JSON string
     */
    @Override
    public String predictWord(String prompts) {
        //
        // Build the command String
        // The general form of commands to the Python script is
        // seqNumber command args...
        //
        ArrayList<String> fullPrompts = new ArrayList<>();
        String promptFormatRegEx = "'(.*?)': \\['(.*?)'\\]";
        Pattern pattern = Pattern.compile(promptFormatRegEx);
        Matcher matcher = pattern.matcher(prompts);

        while (matcher.find()) {
            StringBuilder sb = generatePrefix("predict");
            fullPrompts.add(sb + " {'" + matcher.group(1) + "': ['" + matcher.group(2) + "']}");
        }

        //
        // Invoke and process result
        //
        String results = "";
        for (String fullPrompt : fullPrompts) {
            try {
                AppSession session = pythonService.activate();
                String result = session.invoke(fullPrompt);

                String[] parsed = result.split(" ");
                if (parsed.length != 2) {
                    throw new SSTAFException("Result string does not contain 2 fields. Got '"
                            + result + "' ");
                }
                int expected = requestCount.get() - 1;
                int responseID = Integer.parseInt(parsed[0]);

                if (responseID != expected) {
                    throw new SSTAFException("Response ID [" +
                            responseID + "] did not match expected value [" +
                            expected + "]");
                }

                results += parsed[1];
            } catch(IOException e) {
                throw new SSTAFException("Script invocation failed", e);
            }
        }
        return results;
    }

    @Override
    public ProcessingResult tick(long currentTime_ms) {
        return null;
    }

    @Override
    public ProcessingResult process(HandlerContent o, long l, long l1, Address from, long id, Address respondTo) {

        if (o instanceof PredictWordRequest) {
            PredictWordRequest request = (PredictWordRequest) o;
            String prompts = predictWord(request.getPrompts());
            PredictWordResult pwr = PredictWordResult.builder().prediction(prompts).build();
            return ProcessingResult.of(buildNormalResponse(pwr, id, respondTo));
        } else {
            throw new SSTAFException("Unrecognized command: " + o.toString());
        }
    }
}
