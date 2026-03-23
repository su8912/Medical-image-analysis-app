package com.example.ctscan;


import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;


import com.example.ctscan.ml.Model3;
import com.example.ctscan.ml.ModelCompatible;
import com.example.ctscan.ml.Modelfinal;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class homepage extends AppCompatActivity {

    private final int IMAGE_SIZE = 256;
    private final String[] CLASS_NAMES = {
            "glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"
    };

    ImageView imageView;
    TextView tvPredicted, tvActual, tvConfidence, tvReport;
    Button btnUpload;

    Bitmap bitmapInput;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.home_page);

        imageView = findViewById(R.id.imageView);
        tvPredicted = findViewById(R.id.tv_predicted);
        tvActual = findViewById(R.id.tv_actual);
        tvConfidence = findViewById(R.id.tv_confidence);
        tvReport = findViewById(R.id.tv_report);
        btnUpload = findViewById(R.id.btn_upload);

        btnUpload.setOnClickListener(v -> {
            Intent i = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(i, 101);
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 101 && resultCode == RESULT_OK && data != null) {
            Uri uri = data.getData();
            try {
                Bitmap image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                imageView.setImageBitmap(image);

                bitmapInput = Bitmap.createScaledBitmap(image, IMAGE_SIZE, IMAGE_SIZE, false);
                classifyImage(bitmapInput);
            } catch (IOException e) {
                e.printStackTrace();
                Toast.makeText(this, "Image load error", Toast.LENGTH_SHORT).show();
            }
        }
    }

    private void classifyImage(Bitmap image) {
        try {
            Context context;
            Model3 model = Model3.newInstance(getApplicationContext());

            TensorBuffer inputBuffer = TensorBuffer.createFixedSize(new int[]{1, IMAGE_SIZE, IMAGE_SIZE, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] pixels = new int[IMAGE_SIZE * IMAGE_SIZE];
            image.getPixels(pixels, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            int pixel = 0;
            for (int i = 0; i < IMAGE_SIZE; i++) {
                for (int j = 0; j < IMAGE_SIZE; j++) {
                    int val = pixels[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) / 255.f);
                    byteBuffer.putFloat(((val >> 8) & 0xFF) / 255.f);
                    byteBuffer.putFloat((val & 0xFF) / 255.f);
                }
            }

            inputBuffer.loadBuffer(byteBuffer);

            Model3.Outputs outputs = model.process(inputBuffer);
            TensorBuffer outputBuffer = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputBuffer.getFloatArray();
            int maxIndex = getMaxConfidenceIndex(confidences);

            String predicted = CLASS_NAMES[maxIndex];
            float confidence = confidences[maxIndex];

            tvPredicted.setText("Predicted: " + predicted);
            tvActual.setText("Actual: " + predicted);
            tvConfidence.setText("Confidence: " + String.format("%.2f%%", confidence * 100));

            tvReport.setText(generateReport(confidences));

            model.close();
        } catch (IOException e) {
            Toast.makeText(this, "Model error", Toast.LENGTH_SHORT).show();
            Log.e("ModelLoad", "Failed to load TFLite model: " + e.getMessage());
            Toast.makeText(getApplicationContext(), "Unable to load AI model. Please update the app.", Toast.LENGTH_LONG).show();
            return;
        }
    }

    private int getMaxConfidenceIndex(float[] confidences) {
        int maxIdx = 0;
        float max = confidences[0];
        for (int i = 1; i < confidences.length; i++) {
            if (confidences[i] > max) {
                max = confidences[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private String generateReport(float[] confidences) {
        StringBuilder report = new StringBuilder("Classification Report:\n\n");
        for (int i = 0; i < CLASS_NAMES.length; i++) {
            report.append(CLASS_NAMES[i]).append(": ")
                    .append(String.format("%.2f%%", confidences[i] * 100)).append("\n");
        }
        return report.toString();
    }

}
