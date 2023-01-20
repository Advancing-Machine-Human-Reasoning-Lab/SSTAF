import java.io.*;

public class PythonScriptRunner {
    // filepath to python script
    private String scriptPath;

    public PythonScriptRunner(String scriptPath) {
        this.scriptPath = scriptPath;
    }

    public String runScript(String word) throws IOException, InterruptedException {
        String result = "";

        ProcessBuilder pb = new ProcessBuilder("python", scriptPath, word);
        Process process = pb.start();

        BufferedReader in = new BufferedReader(new InputStreamReader(process.getInputStream()));
        result = in.readLine();

        process.waitFor(); // wait for the script to finish

        return result;
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        String scriptPath = "/raid/home/bez/SSTAF/src/applications/semantic-fluency/predict_word.py";
        PythonScriptRunner runner = new PythonScriptRunner(scriptPath);
        String result = runner.runScript("/raid/home/bez/SSTAF/src/applications/semantic-fluency/prompts.json");
        System.out.println(result);
    }
}
