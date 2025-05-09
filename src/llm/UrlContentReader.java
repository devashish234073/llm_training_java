package llm;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class UrlContentReader {
	public static String read(String url,String filePath) {
		String rawText = "";
		Path path = Paths.get(filePath);
		if(!Files.exists(path)) {
			try {
				URL textUrl = new URL(url);
				StringBuilder stringBuilder = new StringBuilder();
				try (BufferedReader reader = new BufferedReader(new InputStreamReader(textUrl.openStream(), StandardCharsets.UTF_8))) {
					String line;
					while((line=reader.readLine())!=null) {
						stringBuilder.append(line).append(System.lineSeparator());	
					}
					rawText = stringBuilder.toString();
					Files.writeString(path, rawText, StandardCharsets.UTF_8);
					System.out.println("file downloaded successfully");
				} catch(IOException e) {
					System.err.println(e);
				}
			} catch(Exception e) {
				System.err.println(e);
			}
		} else {
			try {
				rawText = Files.readString(path, StandardCharsets.UTF_8);
				System.out.println("file already exists and was read successfully");
			} catch(IOException e) {
				System.out.println(e);
			}
		}
		return rawText;
	}
}
