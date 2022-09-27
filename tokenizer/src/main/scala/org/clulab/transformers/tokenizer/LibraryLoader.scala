package org.clulab.transformers.tokenizer

import java.io.{FileInputStream, FileOutputStream}

object LibraryLoader {
  val label = "[Tokenizer] "

  import java.io.File

  def println(message: String): Unit = {
    System.err.print(label)
    System.err.println(message)
  }

  def loadFromFile(pathname: String): Boolean = {
    try {
      System.load(pathname)
      println("Loading rust interface from " + pathname + "...")
      true
    }
    catch {
      case _: UnsatisfiedLinkError => false
    }
  }

  def loadFromFile(dir: String, libname: String): Boolean = {
    val pathname = dir + File.separator + libname
    loadFromFile(pathname)
  }

  def replaceSuffix(text: String, prev: String, next: String): String =
    if (text.endsWith(prev)) text.substring(0, text.length - prev.length) + next
    else text

  def isMac: Boolean = System.getProperty("os.name").startsWith("Mac ")

  def isApple: Boolean = System.getProperty("os.arch") == "aarch64"

  def mkResourceName(name: String): String = {
    val libName = System.mapLibraryName(name);
    // The Mac reports a libname ending with .dylib, but Java needs .jnilib instead.
    val jniName = replaceSuffix(libName, ".dylib", ".jnilib");
    val resourceName =
        if (!isMac) jniName
        else if (isApple) "apple-" + jniName
        else "intel-" + jniName

    resourceName
  }

  def mkTempFile(name: String): File = {
    val index = name.indexOf('.')
    val prefix = name.substring(0, index)
    val suffix = name.substring(index)
    val tempFile = File.createTempFile(prefix + "-", suffix)

    tempFile
  }

  def load(name: String): Unit = {
    val resourceName = mkResourceName(name)
    val loadedFromPwd = {
      // Try current directory first.
      val location = System.getProperty("user.dir")
      println("Checking " + location + " for " + resourceName + "...")
      loadFromFile(System.getProperty("user.dir"), resourceName)
    }
    val loadedFromHome = !loadedFromPwd && {
      // Try home directory next.
      val location = System.getProperty("user.home")
      println("Checking " + location + " for " + resourceName + "...")
      loadFromFile(location, resourceName);
    }
    val loadedFromResource = !loadedFromPwd && !loadedFromHome && {
      // Attempt to load from the resource via the tmp directory.
      val tempFile = mkTempFile(resourceName)
      println("Extracting resource " + resourceName + " to " + tempFile.getAbsolutePath + "...")

      // Load the jnilib from the JARed resource file, and write it to the temp file.
      // We've anticipated the name and used it for the resource, but that could go wrong.
      val inputStream = getClass.getClassLoader.getResourceAsStream(resourceName)
      val outputStream = new FileOutputStream(tempFile)

      val buf = new Array[Byte](8192)
      while ( {
        val len = inputStream.read(buf)
        if (len > 0) {
          outputStream.write(buf, 0, len)
          true
        }
        else false
      }) {}

      outputStream.flush()
      // Prevent let the OS from removing the temporary file.
      val lock = new FileInputStream(tempFile)
      outputStream.close()

      val loaded = loadFromFile(tempFile.getAbsolutePath)
      lock.close()
      tempFile.delete
      loaded
    }

    if (!(loadedFromPwd || loadedFromHome || loadedFromResource))
      throw new RuntimeException("j4rs could not be loaded!")
  }
}
