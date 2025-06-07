package com.simple.attention;

public class ArrayPrinter {

	public static String prettyPrint2D(float[][] array) {
        if (array == null || array.length == 0) return "[]";
        
        StringBuilder sb = new StringBuilder();
        int cols = array[0].length;
        
        // Determine column widths
        int[] colWidths = new int[cols];
        for (float[] row : array) {
            for (int j = 0; j < cols; j++) {
                colWidths[j] = Math.max(colWidths[j], 
                    String.format("%.4f", row[j]).length());
            }
        }
        
        // Build top border
        sb.append("\n┌");
        for (int j = 0; j < cols; j++) {
            sb.append("─".repeat(colWidths[j] + 2));
            if (j < cols - 1) sb.append("┬");
        }
        sb.append("┐\n");
        
        // Build rows
        for (int i = 0; i < array.length; i++) {
            sb.append("│ ");
            for (int j = 0; j < cols; j++) {
                sb.append(String.format("%" + colWidths[j] + ".4f", array[i][j]));
                sb.append(j < cols - 1 ? " │ " : " │");
            }
            sb.append("\n");
            
            // Add middle border if not last row
            if (i < array.length - 1) {
                sb.append("├");
                for (int j = 0; j < cols; j++) {
                    sb.append("─".repeat(colWidths[j] + 2));
                    if (j < cols - 1) sb.append("┼");
                }
                sb.append("┤\n");
            }
        }
        
        // Build bottom border
        sb.append("└");
        for (int j = 0; j < cols; j++) {
            sb.append("─".repeat(colWidths[j] + 2));
            if (j < cols - 1) sb.append("┴");
        }
        sb.append("┘");
        
        return sb.toString();
    }

    // For 3D arrays (like embeddings)
    public static String prettyPrint3D(float[][][] array) {
        if (array == null || array.length == 0) return "[]";
        
        StringBuilder sb = new StringBuilder();
        
        for (int i = 0; i < array.length; i++) {
            sb.append("Array [").append(i).append("]:\n");
            sb.append(prettyPrint2D(array[i]));
            if (i < array.length - 1) sb.append("\n\n");
        }
        
        return sb.toString();
    }
    
    public static String prettyPrint(float[][] array) {
        StringBuilder sb = new StringBuilder();
        sb.append("[\n");
        
        for (int i = 0; i < array.length; i++) {
            sb.append("  [");
            for (int j = 0; j < array[i].length; j++) {
                sb.append(String.format("%.2f", array[i][j]));
                if (j < array[i].length - 1) {
                    sb.append(", ");
                }
            }
            sb.append("]");
            if (i < array.length - 1) {
                sb.append(",");
            }
            sb.append("\n");
        }
        
        sb.append("]");
        return sb.toString();
    }

    public static String prettyPrint(float[][][] array) {
        StringBuilder sb = new StringBuilder();
        sb.append("[\n");
        
        for (int i = 0; i < array.length; i++) {
            sb.append("  [\n");
            for (int j = 0; j < array[i].length; j++) {
                sb.append("    [");
                for (int k = 0; k < array[i][j].length; k++) {
                    sb.append(String.format("%.2f", array[i][j][k]));
                    if (k < array[i][j].length - 1) {
                        sb.append(", ");
                    }
                }
                sb.append("]");
                if (j < array[i].length - 1) {
                    sb.append(",");
                }
                sb.append("\n");
            }
            sb.append("  ]");
            if (i < array.length - 1) {
                sb.append(",");
            }
            sb.append("\n");
        }
        
        sb.append("]");
        return sb.toString();
    }
    
    public static void printAttentionScores(String prefix, int[] tokenIds, float[][] scores) {
        if (tokenIds == null || scores == null || tokenIds.length != scores.length) {
            throw new IllegalArgumentException("Invalid input dimensions");
        }

        // Determine column widths
        int tokenIdWidth = 8; // Minimum width for "Token ID"
        int scoreWidth = 8;   // Minimum width for scores
        
        for (int id : tokenIds) {
            tokenIdWidth = Math.max(tokenIdWidth, String.valueOf(id).length() + 2);
        }
        
        // Print header
        System.out.printf(prefix+"\n%-" + tokenIdWidth + "s", "Token ID");
        for (int id : tokenIds) {
            System.out.printf("│ %-" + scoreWidth + "s", id);
        }
        System.out.println();

        // Print separator line
        System.out.print(String.format("%-" + tokenIdWidth + "s", "").replace(' ', '─'));
        for (int i = 0; i < tokenIds.length; i++) {
            System.out.print("┼");
            System.out.print(String.format("%-" + (scoreWidth + 1) + "s", "").replace(' ', '─'));
        }
        System.out.println();

        // Print scores
        for (int i = 0; i < tokenIds.length; i++) {
            System.out.printf("%-" + tokenIdWidth + "s", tokenIds[i]);
            for (int j = 0; j < tokenIds.length; j++) {
                System.out.printf("│ %" + scoreWidth + ".4f", scores[i][j]);
            }
            System.out.println();
        }
    }
}