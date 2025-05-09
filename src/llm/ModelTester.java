package llm;

import java.util.Scanner;

public class ModelTester {
	public static void main(String[] args) {
		try {
			Scanner scanner = new Scanner(System.in);
            // Load saved model
			String prompt = null;
            SimpleGPT model = SimpleGPT.loadModel("simple_gpt.bin");
            if (model == null) {
				System.out.println("Failed to load model.");
				return;
			}
            while(!prompt.equals("exit")) {
            	System.out.print("Enter a prompt: ");
                // Generate text
                prompt = scanner.nextLine();
                String generated = model.generate(prompt, 50, 0.7f);
                System.out.println("Generated: " + generated);	
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
	}
}
