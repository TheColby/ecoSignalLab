class Ecosignallab < Formula
  include Language::Python::Virtualenv

  desc "Calibration-aware, multi-channel environmental and architectural acoustics toolkit"
  homepage "https://github.com/TheColby/ecoSignalLab"
  license "MIT"
  head "https://github.com/TheColby/ecoSignalLab.git", branch: "main"

  depends_on "ffmpeg"
  depends_on "libsndfile"
  depends_on "python@3.12"

  def install
    venv = virtualenv_create(libexec, "python3.12")
    venv.pip_install buildpath
    venv.pip_install ".[io,plot,features]"
    bin.install_symlink libexec/"bin/esl"
    man1.install Dir["man/man1/*.1"]
  end

  test do
    assert_match "ecoSignalLab CLI", shell_output("#{bin}/esl --help")
    assert_match "schema_version", shell_output("#{bin}/esl schema 2>&1")
  end

  def caveats
    <<~EOS
      Run `esl doctor` after installation to confirm decode backends and device support.

      If you want optional ML dependencies as well, run:
        #{libexec}/bin/pip install torch datasets huggingface-hub scikit-learn
    EOS
  end
end
