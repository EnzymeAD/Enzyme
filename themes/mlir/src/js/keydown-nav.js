(function($) {
  $(document).ready(function() {
    $('.nav-prev').on( 'click', function() {
      location.href = $(this).attr('href');
    });
    $('.nav-next').on('click', function() {
      location.href = $(this).attr('href');
    });

    $(document).on( 'keydown', function(e) {
      // prev links - left arrow key
      if(e.which == '37') {
        $('.nav-prev').click();
      }

      // next links - right arrow key
      if(e.which == '39') {
        $('.nav-next').click();
      }
    });
  });
})(jQuery);
