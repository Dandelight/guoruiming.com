# `macOS` 上创建自签名根证书

起因是 `liblmgrimpl.dylib` 因为没有签名而导致应用无法启动，找了一圈发现答案：

https://stackoverflow.com/questions/61168329/how-can-i-sign-a-dylib-using-just-a-normal-apple-id-account-no-developer-accou/61176509#61176509

Depends on how you're getting the library.

Building in Xcode

If building in Xcode, you should be able to enable signing and tell it to use Team None and Sign to Run Locally all in the Signing and Capabilities pane of the Project.

Signing an existing dylib

If you're not building it in Xcode and you want to sign a binary you've built or retrieved in some other manner, you're going to need to use codesign, which can be pretty complex.

You can theoretically run codesign using any certificate that has been authorized for code signing, and you can create that with a self-signed certificate, but that is a supreme pain in the neck, and not certain to result in success.

Xcode should automatically create a "Mac Developer" code signing certificate if you have signed in to the developer portal and allowed Xcode to manage signing identifies for you.

You can verify that you have a codesigning identity by using:

security find-identity -v -p codesigning
This will list all of the valid codesigning identities.

Signing the dylib is a matter of using the codesign command:

codesign --force --timestamp --sign <name of certificate> <binary you want to sign>
Using a self-signed code-signing certificate

Note: this is not recommended, but it did work for me.

Start Keychain Access
Choose Keychain Access > Certificate Assistant > Create a Certificate...
Name your certificate something easy to type
Change the Certificate Type to Code Signing
Click Create
Click Continue after reading the warning
Click Done when it completes (and go back to Keychain Access)
Find your newly-created certificate in Keychain Access and double-click on it to get the details
Click the disclosure triangle next to Trust to expose the trust preferences
Set When using this certificate to Always Trust
Now, your self-signed certificate should show up when you run the aforementioned security command to list out the codesigning certificates. If it does, you most likely didn't set the Always Trust or the certificate type to Code Signing.

At this point, you're ready to execute the code signing command, and then you can verify using:

```shell
codesign -vvvv <path to dylib>
```
