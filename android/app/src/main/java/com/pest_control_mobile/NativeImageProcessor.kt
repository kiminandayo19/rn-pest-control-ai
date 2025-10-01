package com.pest_control_mobile

import android.content.res.AssetManager

class NativeImageProcessor {
  companion object {
    init {
      System.loadLibrary("appmodules")
    }

    @JvmStatic
    external fun initAssetManager(assetManager: AssetManager)
  }
}