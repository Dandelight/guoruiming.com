# 绕过网站复制控制的一些方法

```js
// ==UserScript==
// @namespace    http://tampermonkey.net/
// @version      1.1.8
// @description  Lifting copy restrictions on CNKI online reading
// @author       auko
// @supportURL   https://github.com/aukocharlie/my-script
// @include      *://*.cnki.net*/*/Detail*
// @include      *://*/rwt/CNKI/https/*/KXReader/Detail*
// @grant        none
// ==/UserScript==

(function () {
  "use strict";

  var selectText = "";
  document.body.onkeydown = function (e) {
    if (e.ctrlKey && e.keyCode == 67) {
      copy();
      return false;
    }
  };
  document.body.onmouseup = function (e) {
    getSelectText();
  };
  var copytext = document.getElementById("copytext");
  var parent = document.getElementsByClassName("inner")[0];
  if (copytext !== null) parent.removeChild(copytext);

  var proxyBtn = document.createElement("A");

  parent.insertBefore(proxyBtn, parent.children[0]);

  proxyBtn.setAttribute("id", "proxy");
  proxyBtn.innerHTML = "复制";
  document.getElementById("proxy").onclick = function (e) {
    if (document.getElementById("aukoToProxy")) {
      document.getElementById("aukoToProxy").value = selectText;
      document.getElementById("aukoToProxy").select();
    } else {
      var temp = document.createElement("input");
      temp.value = selectText;
      temp.setAttribute("id", "aukoToProxy");
      document.body.appendChild(temp);
      temp.select();
      temp.style.opacity = "0";
    }
    copy();
  };

  function getSelectText() {
    if (document.selection) {
      if (
        document.selection.createRange().text &&
        document.selection.createRange().text !== ""
      ) {
        selectText = document.selection.createRange().text;
      }
    } else {
      if (
        document.getSelection() &&
        document.getSelection().toString() !== ""
      ) {
        selectText = document.getSelection().toString();
      }
    }
  }

  function copy() {
    try {
      if (document.execCommand("Copy", "false", null)) {
        console.log("复制成功！");
      } else {
        console.warn("复制失败！");
      }
    } catch (err) {
      console.warn("复制错误！");
    }
    return false;
  }
})();
```
