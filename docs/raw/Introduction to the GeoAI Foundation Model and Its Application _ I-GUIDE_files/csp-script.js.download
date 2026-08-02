jQuery(document).ready(function($) {
    // When the search icon is clicked...
    $('#csp-search-toggle').on('click', function(e) {
        e.preventDefault();
        // Set display to flex (if needed) and fade in the popup.
        $('#csp-search-popup').css('display', 'flex').hide().fadeIn(200, function() {
            $('#csp-search-input').focus();
        });
    });

    // When clicking outside the inner container, close the popup.
    $(document).mouseup(function(e) {
        var container = $(".csp-search-popup-inner");
        if (!container.is(e.target) && container.has(e.target).length === 0) {
            $('#csp-search-popup').fadeOut(200);
        }
    });

    // When the search form is submitted, close the popup.
    $('#csp-searchform').on('submit', function() {
        $('#csp-search-popup').fadeOut(200);
    });
    
    // Close the popup when the ESC key is pressed.
    $(document).on('keydown', function(e) {
        if (e.key === 'Escape' || e.keyCode === 27) {
            $('#csp-search-popup').fadeOut(200);
        }
    });
});
