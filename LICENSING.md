# Licensing

Camtrak is dual-licensed.

## Open source: GNU AGPL v3

The code in this repository is free software under the
[GNU Affero General Public License v3.0](LICENSE) (or any later version).
You may use, study, modify and share it, for any purpose, at no cost, as long
as you follow the AGPL. In short:

* If you distribute software that includes Camtrak, you must release that
  software's complete source code under the AGPL too.
* If you run a modified Camtrak (or software built on it) as a network
  service, its users must be able to get the source code.

The AGPL fits research, education, personal projects, and open-source
products.

## Commercial license

If you want to ship Camtrak inside a closed-source product, run it in a
proprietary service, or simply prefer not to be bound by the AGPL, a
commercial license lets you do that. It includes:

* use in closed-source products and services, with no obligation to share your code
* priority support and help integrating Camtrak into your system
* a say in the roadmap

Contact **ryan.cuff@icloud.com** with a short description of your use case.

## Third-party components

* `models/vittrack.onnx`: VitTrack model from OpenCV Zoo, Apache License 2.0.
  Optional; used only by the neural second opinion.
* OpenCV and NumPy are dependencies of the Python package, not part of it;
  they are used under their own licenses (Apache 2.0 and BSD).

This summary is not legal advice; the license texts are what apply.
