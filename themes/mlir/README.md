# Hugo Theme Techdoc

The Techdoc is a Hugo Theme for technical documentation.

![The Techdoc screenshot](https://raw.githubusercontent.com/thingsym/hugo-theme-techdoc/master/images/screenshot.png)

## Features

* Modern, Simple layout
* Responsive web design
* Documentation menu (Select Menu style)
* Theme color
* Edit link to documentation repository
* Custom Shortcodes
* Analytics with Google Analytics, Google Tag Manager

## Screenshot

### Theme color

![Theme color](https://raw.githubusercontent.com/thingsym/hugo-theme-techdoc/master/images/screenshot-theme-color.png)

### Menu style

#### Open Menu

![Open Menu](https://raw.githubusercontent.com/thingsym/hugo-theme-techdoc/master/images/screenshot-open-menu.png)

#### Slide Menu

![Slide Menu](https://raw.githubusercontent.com/thingsym/hugo-theme-techdoc/master/images/screenshot-slide-menu.gif)

### Edit link

![Edit link](https://raw.githubusercontent.com/thingsym/hugo-theme-techdoc/master/images/screenshot-edit-link.png)

## Getting Started

### Download Hugo theme

If you have git installed, you can do the following at the command-line-interface within the Hugo directory:

```
cd themes
git clone https://github.com/thingsym/hugo-theme-techdoc.git
```

For more information read [the Hugo documentation](https://gohugo.io/themes/installing-and-using-themes/).

### Configure

You may specify options in config.toml (or config.yaml/config.json) of your site to make use of this theme's features.

For an example of `config.toml`, [config.toml](https://github.com/thingsym/hugo-theme-techdoc/blob/master/exampleSite/config.toml) in exampleSite.

### Directory layout

```
archetypes
  |- default.md
exampleSite
  |- config.toml
  |- ...
images
  |- screenshot.png
  |- tn.png
layouts
  |- _default
  |    |- baseof.html
  |    |- list.html
  |    |- single.html
  |- 404.html
  |- blog
  |    |- li.html
  |    |- list.html
  |    |- single.html
  |    |- summary.html
  |- index.html
  |- partials
  |    |- content-footer.html
  |    |- custom-head.html
  |    |- edit-meta.html
  |    |- edit-page.html
  |    |- footer.html
  |    |- global-menu.html
  |    |- head.html
  |    |- last-updated.html
  |    |- menu.html
  |    |- meta
  |    |    |- chroma.html
  |    |    |- google-analytics-async.html
  |    |    |- google-site-verification.html
  |    |    |- metatag-manager.html
  |    |- notification.html
  |    |- pagination.html
  |    |- powered.html
  |    |- prepend-body.html
  |    |- sidebar-footer.html
  |    |- sidebar.html
  |    |- site-header.html
  |- posts
  |    |- list.html
  |    |- single.html
  |- shortcodes
       |- button.html
       |- panel.html
node_modules
resources
src
  |- scss
  |    |- foundation
  |    |    |- _element.scss
  |    |    |- _normalize.scss
  |    |    |- _reset.scss
  |    |    |- _stack.scss
  |    |- _component.scss
  |    |- _foundation.scss
  |    |- _project.scss
  |    |- _structure.scss
  |    |- _variable.scss
  |    |- chroma.scss
  |    |- theme.scss
  |- js
       |- jquery.backtothetop
       |- functions.js
       |- main.js
static
  |- css
  |    |- chroma.css
  |    |- chroma.min.css
  |    |- theme.css
  |    |- theme.min.css
  |- images
  |- js
    |- bundle.js
.editorconfig
.gitignore
gulpfile.js
LICENSE.md
package.json
README.md
theme.toml
webpack.config.js
```

### Preview site

To preview your site, run Hugo's built-in local server.

```
hugo server -t hugo-theme-techdoc
```

Browse site on http://localhost:1313

## Deploy Site to public_html directory

```
hugo -t hugo-theme-techdoc -d public_html
```

## Development environment

```
cd /path/to/hugo-theme-techdoc
npm install
npm run gulp:watch
```

### Preview exampleSite

```
cd /path/to/dir/themes/hugo-theme-techdoc/exampleSite

hugo server --themesDir ../..
```

Browse site on http://localhost:1313

## Contribution

### Patches and Bug Fixes

Small patches and bug reports can be submitted a issue tracker in Github. Forking on Github is another good way. You can send a pull request.

1. Fork [Hugo Theme Techdoc](http://thingsym.github.io/hugo-theme-techdoc/) from GitHub repository
2. Create a feature branch: git checkout -b my-new-feature
3. Commit your changes: git commit -am 'Add some feature'
4. Push to the branch: git push origin my-new-feature
5. Create new Pull Request

## Changelog

* Version 0.4.0 - 2019.11.02
	* update Sample Document
	* add Theme color
	* add Menu style
	* improve scss
* Version 0.3.0 - 2019.10.06
  * fix archetypes
  * add prepend-body.html for Tag Manager noscript version
  * change class name from menu to global-menu
  * rename partials files
  * fix javascript path for webpack
  * improve development environment
  * move javascript files to src directory
* Version 0.2.2 - 2019.04.27
  * fix Lastmod's and PublishDate's initial value of 0001-01-01
* Version 0.2.1 - 2018.12.07
  * fix scss lint errors
  * change lint from scss-lint to stylelint
  * add published date
  * change the font color of powered by
  * fix link on powered by
* Version 0.2.0 - 2018.11.21
  * add screenshot images
  * add exampleSite
  * fix sub-menu for responsive
  * improve menu and pagination
* Version 0.1.0 - 2018.03.04
  * initial release

## License

Techdoc is licensed under the MIT License.

Techdoc bundles the following third-party resources:

* CSS reset by [normalize.css](https://necolas.github.io/normalize.css/), [MIT](https://opensource.org/licenses/MIT)
* jQuery Plugin [Back to the Top](https://github.com/thingsym/jquery.backtothetop), [MIT](https://opensource.org/licenses/MIT)
* Sass Mixin [Flexbox Grid Mixins](https://thingsym.github.io/flexbox-grid-mixins/), [MIT](https://opensource.org/licenses/MIT)

## Author

[thingsym](https://github.com/thingsym)

Copyright (c) 2017-2019 by thingsym
