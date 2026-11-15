// ============================================================
//  BMF Hackathon (Nov 2026): project list
//
//  To add your project, copy one entry below, fill it in, and
//  open a pull request. Fields:
//    title       (required) project name
//    authors     (required) array of names
//    description (required) one or two sentences
//    site        (optional) your GitHub Pages URL, e.g.
//                https://<your-username>.github.io/<your-repo>/
//    repo        (optional) link to your code on GitHub
//    image       (optional) thumbnail: a full URL, or a file you
//                add to docs/img/ (e.g. "img/my-project.png")
//    tags        (optional) array of short keywords
// ============================================================

const PROJECTS = [
  {
    title: "Double Burden",
    authors: ["Djasahn English"],
    description:
      "Can social indicators alone find Lake County neighborhoods burdened by both pollution and lack of health insurance? An interactive ML explainer with live sliders, a map, and SHAP.",
    site: "https://denglish04.github.io/double-burden-lake-county/",
    repo: "https://github.com/denglish04/double-burden-lake-county",
    image: "img/double_burden_map.png",
    tags: ["EJScreen", "ML", "Python", "Interactive"],
  },

  {
    title: "Redlining × EJScreen",
    authors: ["gibbse-droid"],
    description: "Interactive map overlaying 1930s HOLC redlining maps on EPA EJScreen 2024 indicators.",
    site: "https://gibbse-droid.github.io/redlining-ejscreen/",
    repo: "https://github.com/gibbse-droid/redlining-ejscreen",
    tags: ["EJScreen", "Redlining", "Map"],
  },
  {
    title: "Park Access → Pro Sports",
    authors: ["Natnator08"],
    description: "Tracking park access in low-income neighborhoods and how it connects to youth sports and professional athletes.",
    site: "https://natnator08.github.io/park-access-pro-sports/analysis.html?city=chicago",
    repo: "https://github.com/Natnator08/park-access-pro-sports",
    tags: ["Parks", "Youth sports", "Chicago"],
  },
];
