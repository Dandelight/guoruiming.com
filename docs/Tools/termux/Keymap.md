# Remapping a Bluetooth keyboard on Android

I have a Bluetooth keyboard from my Android device which I am very happy with except that the keyboard itself has no Escape or Tab keys on it. The former isn’t so much of a problem because I can use Ctrl+[ in vim in place of Escape. The latter is if you spend any time in the terminal.

I found a guide on re-mapping the AAndroid keyboard but it needs a little updating for Jelly Bean (the version which I have on my Nexus 7).

Firstly, you’ll need to have a rooted device and access to the adb command. The Android keyboard layouts are stored in the /system/usr/keylayout directory. The defauly keyboard layout which we’ll be modifying is /system/usr/keylayout/Generic.kl.

Connect your device via USB, having made use USB debugging is turned on, and use the adb tool to download the original file to your machine:

```bash
$ adb pull /system/usr/keylayout/Generic.kl
```

This file has a reasonably obvious format. I modified mine to replace Caps Lock with Left Control and re-mapped the Left Control to Tab. My modified layout is available for download and is, like the original, licensed under the Apache 2.0 licence. Obviously you should keep a backup of the original file in case of disaster.

Once you have modified your layout, push it to your device. You cannot write directly to /system since the filesystem is mounted read-only. For the moment, we’ll copy it to the SD card:

```bash
$ adb push Generic.kl /sdcard/Generic.kl
```

Now you’ll need to run some commands on the device either via adb shell on on a terminal emulator running on the device. In either case you’ll need to be the root user. This is usually done via the su command. Then make the /system partition read-write, copy the new keyboard layout and restore the read-only nature of /system

```bash
$ su
# mount -o remount,rw /system
# cp /sdcard/Generic.kl /system/usr/keylayout
# mount -o remount,ro /system
```

Now your keyboard should have your custom mapping.

Refer to: https://rjw57.github.io/notes/technical/android-keyboard/
