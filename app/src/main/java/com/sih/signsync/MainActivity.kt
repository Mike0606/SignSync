package com.sih.signsync

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.sih.signsync.databinding.ActivityMainBinding
import com.google.android.material.bottomnavigation.BottomNavigationView
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.setupActionBarWithNavController
import androidx.navigation.ui.setupWithNavController
import android.Manifest
import android.content.pm.PackageManager

class MainActivity : AppCompatActivity() {

    private lateinit var binding : ActivityMainBinding
    private val req_code = 100
    private val cameraPermissionCode = 101
    private val galleryReqCode = 201
    lateinit var img: ImageView
    lateinit var selectedImage: ImageView
    lateinit var removeCameraPhoto : ImageView
    lateinit var removeGalleryPhoto : ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val buttonCam = binding.buttonCam
        removeCameraPhoto = binding.removeImageCam
        img = binding.imgCapture


        val buttonGallery = binding.buttonPick
        removeGalleryPhoto = binding.removeImageGallery
        selectedImage = binding.imgGallery

        // Check for Camera Permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), cameraPermissionCode)
        }

        buttonCam.setOnClickListener {
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(intent, req_code)
            buttonGallery.isEnabled = false
        }

        removeCameraPhoto.setOnClickListener{
            img.setImageBitmap(null)
            img.visibility = ImageView.GONE
            removeCameraPhoto.visibility = ImageView.GONE
            buttonCam.isEnabled = true
            buttonGallery.isEnabled = true
        }

        buttonGallery.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, galleryReqCode)
            buttonCam.isEnabled = false
        }
        removeGalleryPhoto.setOnClickListener{
            selectedImage.setImageBitmap(null)
            selectedImage.visibility = ImageView.GONE
            removeCameraPhoto.visibility = ImageView.GONE
            buttonCam.isEnabled = true
            buttonGallery.isEnabled = true
        }
    }

    // Handle the result of the permission request
    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == cameraPermissionCode) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission granted, you can now use the camera
            } else {
                Toast.makeText(this, "Access to camera denied", Toast.LENGTH_SHORT).show()
            }
        }
    }

    // Handle the result of camera and gallery intents
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == RESULT_OK) {
            when(requestCode) {
                req_code -> {
                    val imageBitmap = data?.extras?.get("data") as Bitmap
                    img.setImageBitmap(imageBitmap)
                    img.visibility = ImageView.VISIBLE
                    removeCameraPhoto.visibility = ImageView.VISIBLE
                }
                galleryReqCode -> {
                    val selectedImg: Uri? = data?.data
                    selectedImage.setImageURI(selectedImg)
                    selectedImage.visibility = ImageView.VISIBLE
                    removeGalleryPhoto.visibility = ImageView.VISIBLE
                }
            }
        }
    }
}
