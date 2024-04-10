
/*
 * Copyright (c) 2023 Frenify
 * Author: Frenify
 * This file is made for CURRENT TEMPLATE
*/


$.fn.frenifyMoveCursorToEnd = function () {
	"use strict";
	this.focus();
	var $thisVal = this.val();
	this.val('').val($thisVal);
	return this;
};
var FrenifyTechWaveTime = new Date();

(function ($) {
	"use strict";

	var TechwaveSelectedCount = 0;

	var TechwaveFeedFilterLoading = false;
	var TechwaveModelFilterLoading = false;

	var TechwaveInputText = '';
	var TechwaveUserQuestion = '';


	var TechwaveUserMessageCount = 0;


	var FrenifyTechWave = {

		init: function () {
			this.marquee();
			this.tooltip();
			this.fontDialog();
			this.modelTabs();
			this.bookmark();
			this.contactForm();
			this.negativePrompt();
			this.imageGenerationSidebar();
			this.rangeSlider();
			this.quantity();
			this.selectModel();
			this.anchor();
			this.aiChatBot__chat();
			this.aiChatBotOptions();
			this.aiChatBotTextareaHeight();
			this.billingProgress();
			this.inputFileOnChange();
			this.optionsList();
			this.pricingTab();
			this.feedFilters();
			this.report();
			this.follow();
			this.copyLink();
			this.galleryIsotope();
			this.imageLightbox();
			this.like();
			this.accordion();
			this.search();
			this.animatedText();
			this.movingSubMenuForLeftPanel();
			this.panelResize();
			this.navBarItems();
			this.redetectFullScreen();
			this.fullSCreen();
			this.navSubMenu();
			this.imgToSVG();
			this.BgImg();
			this.popupMobile();
			this.upload();
		},

		upload: function () {
			$("#uploadBtn").off().on('click', function () {
				var files = $('#filelist')[0].files;

				var fileData = new FormData();
				// Add each file to the FormData object
				for (var i = 0; i < files.length; i++) {
					fileData.append('files[]', files[i]);
				}

				$(".techwave_fn_preloader").css("display", "flex");

				$.ajax({
					url: `/uploadDocuments`,
					method: 'POST',
					data: fileData,
					processData: false,
					contentType: false,
					success: function (result) {
						if (result.success === "ok") {
							$(".upload_content").empty();
							$(".upload_content").append('<span class="title">Drag & Drop a PDF</span><span class="desc">Supports PDF</span>');
							$(".fn__upload").removeClass("has_file");
							var activeTab = $(".techwave_fn_models .fn__tabs a.active");
							activeTab.removeClass("active");
							activeTab.trigger("click");

							$(".alert").html('Succeed');
						}
						else $(".alert").html('Failed');

						$(".techwave_fn_preloader").css("display", "none");

					},
					error: function (xhr, status, error) {
						console.error('Error occurred: ', error);
						alert("Error occurred!");
						$(".techwave_fn_preloader").css("display", "none");
						$(".alert").html('Failed');
					}
				});


			});
		},

		marquee: function () {
			$(".TickerNews .marquee").each(function () {
				var e = $(this);
				if (!e.hasClass('ready')) {
					e.addClass('ready').marquee({
						duplicated: true,
						duration: parseInt(e.data('speed')) * 1000,
						delayBeforeStart: 0,
						direction: 'left',
						//						pauseOnHover: true,
						startVisible: true
					});
				}
			});
		},

		popupMobile: function () {
			if (window.matchMedia('(max-width: 767px)').matches) {
				var wrapperW = $('.techwave_fn_wrapper').width();
				var padding = 10;
				var maxWidth = 300;
				$('.item__popup,.fn__nav_bar .item_popup').each(function () {
					var element = $(this);
					var parent = element.parent();
					var width = wrapperW - 2 * padding;
					var normal = Math.min(width, maxWidth);
					var leftOffset = parent.offset().left;
					var left = padding - leftOffset + (width - normal) / 2;

					var right = 'auto';

					if (element.data('position') === 'right') {
						if (leftOffset + parent.width() > normal) {
							left = 'auto';
							right = 0;
						}
					} else {
						if ((leftOffset + normal) < width) {
							left = 0;
						}
					}

					element.css({ maxWidth: normal, width: normal, left: left, right: right });
				});
			} else {
				$('.fn__nav_bar .item_popup,.item__popup').attr('style', '');
			}
		},

		tooltip: function () {
			$('body').on('mouseover mouseenter', '.fn__tooltip', function () {
				var element = $(this);
				var position = element.attr('data-position');
				if (typeof position === 'undefined' || position === true) {
					position = ['top', 'bottom', 'right', 'left'];
				}
				var options = {
					contentAsHTML: 'true',
					maxWidth: 300,
					animationDuration: 0,
					animation: 'fade', // 'fade', 'grow', 'swing', 'slide', 'fall'
					delay: 0,
					theme: 'tooltipster-techwave',
					side: position
				};
				if (element.hasClass('menu__item')) {
					if (!$('html').hasClass('panel-opened')) {
						element.tooltipster(options).tooltipster('hide');
						return;
					}
				}
				element.tooltipster(options);
				element.tooltipster('show');
			});
			//			$( ".fn__tooltip" ).each(function(){
			//				$(this).tooltipster({
			//					theme: 'tooltipster-techwave',
			//					animation: 'fade', // 'fade', 'grow', 'swing', 'slide', 'fall'
			//					side: 'bottom',
			//					maxWidth: 300,
			//					animationDuration: 0,
			//					delay: 0,
			//					trigger: 'click'
			//				});	
			//			});

		},

		fontDialog: function () {
			var dialog = $('.techwave_fn_font');
			$('.font__trigger').off().on('click', function () {
				dialog.addClass('opened');
				return false;
			});


			dialog.find('.font__closer').off().on('click', function () {
				dialog.removeClass('opened');
				return false;
			});
			dialog.find('.font__closer_link').off().on('click', function () {
				dialog.removeClass('opened');
				return false;
			});
			dialog.find('.apply').off().on('click', function () {
				$('.fn__chat_font_size_style').remove();
				$('body').append('<style type="text/css" class="fn__chat_font_size_style">frenify_typing h3,.fn__chatbot .chat{font-size: ' + $('#font_size').find(":selected").val() + 'px;}</style>');
				dialog.removeClass('opened');
				return false;
			});
		},

		modelTabs: function () {
			// tab filter
			$('.techwave_fn_models .fn__tabs a').off().on('click', function () {
				var element = $(this);
				if (!element.hasClass('active') && !TechwaveModelFilterLoading) {
					TechwaveModelFilterLoading = true;
					element.siblings().removeClass('active');
					element.addClass('active');
					var parent = element.closest('.techwave_fn_models');
					parent.find('.models__results').addClass('loading');

					// do your ajax here
					// you have to get new models with filter (if you want to change also filter) of selected tab via ajax
					// ....
					// ....
					// after ajax ends remove setTimeout (it was added just for HTML)

					setTimeout(function () {
						// here you have to insert your models into the filter's content and feed content
						parent.find('.models__results').removeClass('loading');
						parent.find('.tab__item.active').removeClass('active');
						$(element.attr('href')).addClass('active');
						TechwaveModelFilterLoading = false;

					}, 1500);

				}
				return false;
			});
		},

		contactForm: function () {
			$("#send_message").on('click', function () {
				var name = $(".fn_contact_form #name").val();
				var email = $(".fn_contact_form #email").val();
				var tel = $(".fn_contact_form #tel").val();
				var message = $(".fn_contact_form #message").val();
				var success = $(".fn_contact_form .returnmessage").data('success');

				$(".fn_contact_form .returnmessage").empty(); //To empty previous error/success message.
				//checking for blank fields	
				if (name === '' || email === '' || message === '') {
					$('.fn_contact_form .empty_notice').slideDown(500).delay(2000).slideUp(500);
				}
				else {
					// Returns successful data submission message when the entered information is stored in database.
					$.post("modal/contact.php", { ajax_name: name, ajax_email: email, ajax_message: message, ajax_tel: tel }, function (data) {

						$(".fn_contact_form .returnmessage").append(data);//Append returned message to message paragraph


						if ($(".fn_contact_form .returnmessage span.contact_error").length) {
							$(".fn_contact_form .returnmessage").slideDown(500).delay(2000).slideUp(500);
						} else {
							$(".fn_contact_form .returnmessage").append("<span class='contact_success'>" + success + "</span>");
							$(".fn_contact_form .returnmessage").slideDown(500).delay(4000).slideUp(500);
						}

						if (data === "") {
							$("#fn_contact_form")[0].reset();//To reset form fields on success
						}

					});
				}
				return false;
			});
		},

		negativePrompt: function () {
			$('#negative_prompt').on('change', function () {
				if (this.checked) {
					$('.techwave_fn_image_generation_page .exclude_area').slideDown(200);
				} else {
					$('.techwave_fn_image_generation_page .exclude_area').slideUp(200);
				}
			});
		},

		imageGenerationSidebar: function () {
			$('.techwave_fn_image_generation_page .sidebar__trigger').off().on('click', function () {
				$('.techwave_fn_wrapper').toggleClass('fn__has_sidebar');
				return false;
			});
		},

		rangeSlider: function () {
			$('.fn__range').each(function () {
				var element = $(this),
					input = element.find('input'),
					val = input.val(),
					output = element.find('.value'),
					min = input.attr('min'),
					max = input.attr('max'),
					slider = element.find('.slider');
				slider.css({ width: (val * (min * 100) / max) + '%' });
				input.on('input', function () {
					val = $(this).val();
					output.text(val);
					slider.css({ width: (val * (min * 100) / max) + '%' });
				});
			});
		},

		quantity: function () {
			$('.fn__quantity .increase').off().on('click', function () {
				var parent = $(this).closest('.fn__quantity');
				var input = parent.find('input');
				var max = parseInt(input.attr('max'), 10);
				var value = parseInt(input.val(), 10);
				value = isNaN(value) ? 0 : value;
				if (max === value) {
					return false;
				}
				value++;
				input.val(value);
				return false;
			});
			$('.fn__quantity .decrease').off().on('click', function () {
				var parent = $(this).closest('.fn__quantity');
				var input = parent.find('input');
				var value = parseInt(input.val(), 10);
				var min = parseInt(input.attr('min'), 10);
				value = isNaN(value) ? 0 : value;
				if (min === value) {
					return false;
				}
				value--;
				input.val(value);

				return false;
			});
		},

		selectModel: function () {
			$('.fn__select_model .model_open').off().on('click', function () {
				$(this).closest('.fn__select_model').toggleClass('opened');
				return false;
			});

			$(window).on('click', function () {
				$('.fn__select_model').removeClass('opened');
			});

			$('.fn__select_model .all_models').on('click', function (e) {
				e.stopPropagation();
			});
		},

		anchor: function () {
			$('.techwave_fn_doc_page .docsidebar li.menu-item-has-children > a').off().on('click', function () {
				$(this).siblings('ul').slideToggle();
				return false;
			});
			if ($().onePageNav) {
				$('.techwave_fn_doc_page .docsidebar > ul').onePageNav();
			}
		},

		aiChatBot__chat: function () {
			if ($('#fn__chat_textarea').length && !$('.techwave_fn_intro').length) {
				$("#fn__chat_textarea").focus();
			}
			$("#fn__chat_textarea").keypress(function (e) {
				var code = (e.keyCode ? e.keyCode : e.which);
				if (code === 13 && e.shiftKey) {

				} else if (code === 13) {
					$('.fn__chat_comment button').trigger('click');
					return false;
				}
			});
			$('.fn__chat_comment button').off().on('click', function () {
				var button = $(this);
				var textarea = $('#fn__chat_textarea');
				var text = textarea.val();
				if (text === '' || button.hasClass('disabled')) {
					return;
				} else {
					text = text.replace(/\n\r?/g, '<br />');
					TechwaveUserQuestion = text;
					var activeChatItem = $('.fn__chatbot .chat__item.active');
					var newText = '<div class="chat__box your__chat"><div class="author"><span>You</span></div><div class="chat"><p>' + text + '</p></div></div>';
					$('.fn__chat_comment').removeClass('neww');
					if (activeChatItem.attr('id') === 'chat0') {
						activeChatItem.removeClass('active');
						$('.fn__new_chat_link').removeClass('active');
						var id = $('.fn__chatbot .chat__item').length;
						$('.fn__chatbot .chat__list').append('<div class="chat__item active" id="chat' + id + '">' + newText + '</div>');
						var newChatToRightPanel = '<li class="group__item"><div class="fn__chat_link active" href="#chat' + id + '"><span class="text">New Chat</span><input type="text" value="New Chat"><span class="options"><button class="trigger"><span></span></button><span class="options__popup"><span class="options__list"><button class="edit">Edit</button><button class="delete">Delete</button></span></span></span><span class="save_options"><button class="save"><img src="svg/check.svg" alt="" class="fn__svg"></button><button class="cancel"><img src="svg/close.svg" alt="" class="fn__svg"></button></span></div></li>';
						if ($('.fn__chatbot .chat__group.new').length) {
							$('.fn__chatbot .chat__group.new ul').append(newChatToRightPanel);
						} else {
							$('.fn__chatbot .sidebar_content').prepend('<div class="chat__group"><h2 class="group__title">Today</h2><ul class="group__list">' + newChatToRightPanel + '</ul></div>');
						}
						FrenifyTechWave.imgToSVG();
						FrenifyTechWave.aiChatBotOptions();
					} else {
						activeChatItem.append(newText);
					}
					textarea.val('');
					textarea.siblings('.fn__hidden_textarea').val('');
					FrenifyTechWave.aiChatBotTextareaHeight();

					if ($('.techwave_fn_intro').length) {
						$("html, body").animate({ scrollTop: textarea.offset().top - $(window).height() + 100 });
					} else {
						$("html, body").animate({ scrollTop: $(document).height() - $(window).height() });
					}
					textarea.frenifyMoveCursorToEnd();
				}

				// do your ajax here to answer the chat via AI

				// you can remove frenify chat bot here
				FrenifyTechWave.tdmchat();

				// don't remove this
				return false;
			});
		},

		tdmchat: function () {
			console.log(TechwaveUserQuestion);
			$.ajaxSetup({
				headers:
					{ 'X-CSRF-TOKEN': $('meta[name="csrf-token"]').attr('content') }
			});

			var title = $("#page_title").val();
			var sendData = {
				"prompt": TechwaveUserQuestion,
				"title": title
			};

			$.ajax({
				url: '/query',
				method: 'POST',
				contentType: 'application/json',
				data: JSON.stringify(sendData),
				success: function (result) {
					if (result.success === "ok") {
						console.log(result.response);
						$('.fn__chat_comment button').addClass('disabled');
						setTimeout(function () {
							$('.fn__chatbot .chat__item.active').append('<div class="chat__box bot__chat"><div class="author"><span>Real Estate Agent</span></div><div class="chat"><frenify_typing><h3><span>Thinking...</frenify></h3></div></div>');
							if ($('.techwave_fn_intro').length) {
								$("html, body").animate({ scrollTop: $('#fn__chat_textarea').offset().top - $(window).height() + 100 });
							} else {
								$("html, body").animate({ scrollTop: $(document).height() - $(window).height() });
							}
						}, 100);
						setTimeout(function () {
							$('.fn__chatbot .chat__item.active .chat__box.bot__chat:last-child .chat').html(result.response);
							$('.fn__chat_comment button').removeClass('disabled');
							if ($('.techwave_fn_intro').length) {
								$("html, body").animate({ scrollTop: $('#fn__chat_textarea').offset().top - $(window).height() + 100 });
							} else {
								$("html, body").animate({ scrollTop: $(document).height() - $(window).height() });
							}
						}, 2000);


						$('.sidebar_content').html('');
						for (var i = 0; i < 3; i++) {
							$('.sidebar_content').append('<img src="' + result.images[i] + '">');
						}
					}
					else console.error("An error occurred...");
				},
				error: function (xhr, status, error) {
					console.error('Error occurred: ', error);
					alert("Error occurred!")
				}
			});
		},
		frenifyChat: function () {
			var botText = '';
			var append = true;
			var commands = '';

			TechwaveUserMessageCount = $('.fn__chatbot .chat__item.active .chat__box').length;

			// all avaliable commands
			var allCommands = {
				welcome: {
					type: 'text',
					description: 'welcome message',
					text: '<p>Frenify was founded in 2017. The company began working with the first customers, giving them the opportunity to purchase high-quality HTML templates.</p><p>The company’s products began to grow in terms of complexity and aesthetics. Frenify currently has a wide range of HTML templates, WordPress themes, WordPress plugins, Photoshop projects; paid and absolutely free products.</p><p>Design projects are unique and aesthetically pleasing based on customer requirements. Visit our website to get acquainted with our products. Thank you so much for being with us.</p>',
					append: true
				},
				about: {
					type: 'text',
					description: 'some information about the Frenify team',
					text: '<p>Frenify was founded in 2017. The company began working with the first customers, giving them the opportunity to purchase high-quality HTML templates.</p><p>The company’s products began to grow in terms of complexity and aesthetics. Frenify currently has a wide range of HTML templates, WordPress themes, WordPress plugins, Photoshop projects; paid and absolutely free products.</p><p>Design projects are unique and aesthetically pleasing based on customer requirements. Visit our website to get acquainted with our products. Thank you so much for being with us.</p>',
					append: true
				},
				website: {
					type: 'url',
					description: 'go to our official website',
					append: false,
					url: 'https://frenify.com/'
				},
				free: {
					type: 'url',
					description: 'get PSD files of premium themes for free',
					append: false,
					url: 'https://frenify.com/freebies/'
				},
				doc: {
					type: 'url',
					description: 'visit online documentation for TechWave HTML template',
					append: false,
					url: 'https://frenify.com/work/envato/frenify/html/techwave/doc'
				},
				support: {
					type: 'url',
					description: 'if you have any questions regarding TechWave HTML template feel free and contact us by this command',
					append: false,
					url: 'https://themeforest.net/item/techwave-ai-html-dashboard-for-image-generation-chat-bot/46197058/support/contact'
				},
				purchase: {
					type: 'url',
					description: 'open the template description page on themeforest to purchase it',
					append: false,
					url: 'https://themeforest.net/item/techwave-ai-html-dashboard-for-image-generation-chat-bot/46197058'
				},
				youtube: {
					type: 'url',
					description: 'visit our youtube channel with video guides on our themes and templates',
					append: false,
					url: 'https://www.youtube.com/@frenifyteam/videos'
				},
				pass: {
					type: 'password',
					description: 'if you want to get strong password I can generate it for you, write <frenify_main>/pass 20</frenify_main> to get a 20 character password',
					append: true,
				},
				joke: {
					type: 'joke',
					description: 'I can cheer you up by telling a joke',
					append: true,
				},
				time: {
					type: 'time',
					description: 'display current time',
					append: true,
				},
				clear: {
					type: 'clear',
					description: 'to clear current chat',
					append: false,
				},
				commands: {
					type: 'commands',
					description: 'to list all available commands',
					append: true,
				},
			};



			// collect all commands
			commands = '<ul>';
			$.each(allCommands, function (key, value) {
				commands += '<li><frenify_main>/' + key + '</frenify_main> - ' + value.description + '</li>';
			});
			commands += '</ul>';

			// welcome text (first message of the bot)
			var welcomeText = '<p>Hello.</p><p>I am Frenify Bot. After purchasing the template, you can delete me easily. I understand some commands. You just select one of the commands and write here.</p><p>Here is the list of commands:</p>' + commands + '<p>We are trying for you. We try to make unique themes and templates with excellent functionality and excellent design.</p>';

			allCommands.welcome.text = welcomeText;


			var allJokes = [
				'What did one pirate say to the other when he beat him at chess?<>Checkmatey.',
				'I burned 2000 calories today<>I left my food in the oven for too long.',
				'I startled my next-door neighbor with my new electric power tool. <>I had to calm him down by saying “Don’t worry, this is just a drill!”',
				'I broke my arm in two places. <>My doctor told me to stop going to those places.',
				'I quit my job at the coffee shop the other day. <>It was just the same old grind over and over.',
				'I never buy anything that has Velcro with it...<>it’s a total rip-off.',
				'I used to work at a soft drink can crushing company...<>it was soda pressing.',
				'I wondered why the frisbee kept on getting bigger. <>Then it hit me.',
				'I was going to tell you a fighting joke...<>but I forgot the punch line.',
				'What is the most groundbreaking invention of all time? <>The shovel.',
				'I’m starting my new job at a restaurant next week. <>I can’t wait.',
				'I visited a weight loss website...<>they told me I have to have cookies disabled.',
				'Did you hear about the famous Italian chef that recently died? <>He pasta way.',
				'Broken guitar for sale<>no strings attached.',
				'I could never be a plumber<>it’s too hard watching your life’s work go down the drain.',
				'I cut my finger slicing cheese the other day...<>but I think I may have grater problems than that.',
				'What time did you go to the dentist yesterday?<>Tooth-hurty.',
				'What kind of music do astronauts listen to?<>Neptunes.',
				'Rest in peace, boiled water. <>You will be mist.',
				'What is the only concert in the world that costs 45 cents? <>50 Cent, featuring Nickelback.',
				'It’s not a dad bod<> it’s a father figure.',
				'My wife recently went on a tropical food diet and now our house is full of this stuff. <>It’s enough to make a mango crazy.',
				'What do you call Santa’s little helpers? <>Subordinate clauses.',
				'Want to hear a construction joke? <>Sorry, I’m still working on it.',
				'What’s the difference between a hippo and a zippo? <>One is extremely big and heavy, and the other is a little lighter.',
				'I burnt my Hawaiian pizza today in the oven, <>I should have cooked it on aloha temperature.',
				'Anyone can be buried when they die<>but if you want to be cremated then you have to urn it.',
				'Where did Captain Hook get his hook? <>From the second-hand store.',
				'I am such a good singer that people always ask me to sing solo<>solo that they can’t hear me.',
				'I am such a good singer that people ask me to sing tenor<>tenor twelve miles away.',
				'Occasionally to relax I just like to tuck my knees into my chest and lean forward.<> That’s just how I roll.',
				'What did the glass of wine say to the glass of beer? Nothing. <>They barley knew each other.',
				'I’ve never trusted stairs. <>They are always up to something.',
				'Why did Shakespeare’s wife leave him? <>She got sick of all the drama.',
				'I just bought a dictionary but all of the pages are blank. <>I have no words to describe how mad I am.',
				'If you want to get a job at the moisturizer factory... <>you’re going to have to apply daily.',
				'I don’t know what’s going to happen next year. <>It’s probably because I don’t have 2020 vision.',
				'Want to hear a joke about going to the bathroom? <>Urine for a treat.',
				'I couldn’t figure out how to use the seat belt. <>Then it just clicked.',
				'I got an email the other day teaching me how to read maps backwards<>turns out it was just spam.',
				'I’m reading a book about anti-gravity.<> It’s impossible to put down!',
				'You’re American when you go into the bathroom, and you’re American when you come out, but do you know what you are while you’re in there?<> European.',
				'Did you know the first French fries weren’t actually cooked in France?<> They were cooked in Greece.',
				'Want to hear a joke about a piece of paper? Never mind... <>it’s tearable.',
				'I just watched a documentary about beavers. <>It was the best dam show I ever saw!',
				'If you see a robbery at an Apple Store what re you?<> An iWitness?',
				'Spring is here! <>I got so excited I wet my plants!',
				'What’s Forrest Gump’s password?<> 1forrest1',
				'Why did the Clydesdale give the pony a glass of water? <>Because he was a little horse!',
				'CASHIER: "Would you like the milk in a bag, sir?" <>DAD: "No, just leave it in the carton!’”',
				'Did you hear about the guy who invented Lifesavers? <>They say he made a mint.',
				'I bought some shoes from a drug dealer.<> I don’t know what he laced them with, but I was tripping all day!',
				'Why do chicken coops only have two doors?<> Because if they had four, they would be chicken sedans!',
				'How do you make a Kleenex dance? <>Put a little boogie in it!',
				'A termite walks into a bar and asks<>"Is the bar tender here?"',
				'Why did the invisible man turn down the job offer?<> He couldn’t see himself doing it.',
				'I used to have a job at a calendar factory <>but I got the sack because I took a couple of days off.',
				'A woman is on trial for beating her husband to death with his guitar collection. Judge says, "First offender?" <>She says, "No, first a Gibson! Then a Fender!”',
				'How do you make holy water?<> You boil the hell out of it.',
				'I had a dream that I was a muffler last night.<> I woke up exhausted!',
				'Did you hear about the circus fire?<> It was in tents!',
				'Don’t trust atoms.<> They make up everything!',
				'How many tickles does it take to make an octopus laugh? <>Ten-tickles.',
				'I’m only familiar with 25 letters in the English language.<> I don’t know why.',
				'Why did the cow in the pasture get promoted at work?<> Because he is OUT-STANDING in his field!',
				'What do prisoners use to call each other?<> Cell phones.',
				'Why couldn’t the bike standup by itself? <>It was two tired.',
				'Who was the fattest knight at King Arthur’s round table?<> Sir Cumference.',
				'Did you see they made round bails of hay illegal in Wisconsin? <>It’s because the cows weren’t getting a square meal.',
				'You know what the loudest pet you can get is?<> A trumpet.',
				'What do you get when you cross a snowman with a vampire?<> Frostbite.',
				'What do you call a deer with no eyes?<> No idea!',
				'Can February March? <>No, but April May!',
				'What do you call a lonely cheese? <>Provolone.',
				'Why can’t you hear a pterodactyl go to the bathroom?<> Because the pee is silent.',
				'What did the buffalo say to his son when he dropped him off at school?<> Bison.',
				'What do you call someone with no body and no nose? <>Nobody knows.',
				'You heard of that new band 1023MB? <>They’re good but they haven’t got a gig yet.',
				'Why did the crab never share?<> Because he’s shellfish.',
				'How do you get a squirrel to like you? <>Act like a nut.',
				'Why don’t eggs tell jokes? <>They’d crack each other up.',
				'Why can’t a nose be 12 inches long? <>Because then it would be a foot.',
				'Did you hear the rumor about butter? <>Well, I’m not going to spread it!',
				'I made a pencil with two erasers. <>It was pointless.',
				'I used to hate facial hair...<>but then it grew on me.',
				'I decided to sell my vacuum cleaner—<>it was just gathering dust!',
				'I had a neck brace fitted years ago<> and I’ve never looked back since.',
				'You know, people say they pick their nose,<> but I feel like I was just born with mine.',
				'What do you call an elephant that doesn’t matter?<> An irrelephant.',
				'What do you get from a pampered cow? <>Spoiled milk.',
				'It’s inappropriate to make a ’dad joke’ if you’re not a dad.<> It’s a faux pa.',
				'How do lawyers say goodbye? <>Sue ya later!',
				'Wanna hear a joke about paper? <>Never mind—it’s tearable.',
				'What’s the best way to watch a fly fishing tournament? <>Live stream.',
				'I could tell a joke about pizza,<> but it’s a little cheesy.',
				'When does a joke become a dad joke?<> When it becomes apparent.',
				'What’s an astronaut’s favorite part of a computer? <>The space bar.',
				'What did the shy pebble wish for?<>That she was a little boulder.',
				'I’m tired of following my dreams. <>I’m just going to ask them where they are going and meet up with them later.',
				'Did you hear about the guy whose whole left side was cut off? <>He’s all right now.',
				'Why didn’t the skeleton cross the road? <>Because he had no guts.',
				'What did one nut say as he chased another nut? <> I’m a cashew!',
				'Chances are if you’ve seen one shopping center...<> you’ve seen a mall.',
				'I knew I shouldn’t steal a mixer from work...<>but it was a whisk I was willing to take.',
				'How come the stadium got hot after the game? <>Because all of the fans left.',
				'Why was it called the dark ages? <>Because of all the knights.',
				'Why did the tomato blush? <>Because it saw the salad dressing.',
				'Did you hear the joke about the wandering nun? <>She was a roman catholic.',
				'What creature is smarter than a talking parrot? <>A spelling bee.',
				'I’ll tell you what often gets over looked...<> garden fences.',
				'Why did the kid cross the playground? <>To get to the other slide.',
				'Why do birds fly south for the winter?<> Because it’s too far to walk.',
				'What is a centipedes’s favorite Beatle song? <> I want to hold your hand, hand, hand, hand...',
				'My first time using an elevator was an uplifting experience. <>The second time let me down.',
				'To be Frank...<> I’d have to change my name.',
				'Slept like a log last night … <>woke up in the fireplace.',
				'Why does a Moon-rock taste better than an Earth-rock? <>Because it’s a little meteor.',
				'How many South Americans does it take to change a lightbulb?<> A Brazilian',
				'I don’t trust stairs.<> They’re always up to something.',
				'A police officer caught two kids playing with a firework and a car battery.<> He charged one and let the other one off.',
				'What is the difference between ignorance and apathy?<>I don’t know and I don’t care.',
				'I went to a Foo Fighters Concert once... <>It was Everlong...',
				'Some people eat light bulbs. <>They say it’s a nice light snack.',
				'What do you get hanging from Apple trees? <> Sore arms.',
				'Last night me and my girlfriend watched three DVDs back to back.<> Luckily I was the one facing the TV.',
				'I got a reversible jacket for Christmas,<> I can’t wait to see how it turns out.',
				'What did Romans use to cut pizza before the rolling cutter was invented? <>Lil Caesars',
				'My pet mouse ’Elvis’ died last night. <>He was caught in a trap..',
				'Never take advice from electrons. <>They are always negative.',
				'Why are oranges the smartest fruit? <>Because they are made to concentrate. ',
				'What did the beaver say to the tree? <>It’s been nice gnawing you.',
				'How do you fix a damaged jack-o-lantern?<> You use a pumpkin patch.',
				'What did the late tomato say to the early tomato? <>I’ll ketch up',
				'I have kleptomania...<>when it gets bad, I take something for it.',
				'I used to be addicted to soap...<> but I’m clean now.',
				'When is a door not a door?<> When it’s ajar.',
				'I made a belt out of watches once...<> It was a waist of time.',
				'This furniture store keeps emailing me,<> all I wanted was one night stand!',
				'How do you find Will Smith in the snow?<>  Look for fresh prints.',
				'I just read a book about Stockholm syndrome.<> It was pretty bad at first, but by the end I liked it.',
				'Why do trees seem suspicious on sunny days? <>Dunno, they’re just a bit shady.',
				'If at first you don’t succeed<> sky diving is not for you!',
				'What kind of music do mummy’s like?<>Rap',
				'A book just fell on my head. <>I only have my shelf to blame.',
				'What did the dog say to the two trees? <>Bark bark.',
				'If a child refuses to sleep during nap time...<> are they guilty of resisting a rest?',
				'Have you ever heard of a music group called Cellophane?<> They mostly wrap.',
				'What did the mountain climber name his son?<>Cliff.',
				'Why should you never trust a pig with a secret?<> Because it’s bound to squeal.',
				'Why are mummys scared of vacation?<> They’re afraid to unwind.',
				'Whiteboards ...<> are remarkable.',
				'What kind of dinosaur loves to sleep?<>A stega-snore-us.',
				'What kind of tree fits in your hand?<> A palm tree!',
				'I used to be addicted to the hokey pokey<> but I turned myself around.',
				'How many tickles does it take to tickle an octopus?<> Ten-tickles!',
				'What musical instrument is found in the bathroom?<> A tuba toothpaste.',
				'My boss told me to attach two pieces of wood together... <>I totally nailed it!',
				'What was the pumpkin’s favorite sport?<>Squash.',
				'What do you call corn that joins the army?<> Kernel.',
				'I’ve been trying to come up with a dad joke about momentum <>but I just can’t seem to get it going.',
				'Why don’t sharks eat clowns? <> Because they taste funny.',
				'Just read a few facts about frogs.<> They were ribbiting.',
				'Why didn’t the melons get married?<>Because they cantaloupe.',
				'What’s a computer’s favorite snack?<>Microchips!',
				'Why was the robot so tired after his road trip?<>He had a hard drive.',
				'Why did the computer have no money left?<>Someone cleaned out its cache!',
				'I’m not anti-social. <>I’m just not user friendly.',
				'Why did the computer get cold?<>Because it forgot to close windows.',
				'What is an astronaut’s favorite key on a keyboard?<>The space bar!',
				'What’s the difference between a computer salesman and a used-car salesman?<>The used-car salesman KNOWS when he’s lying.',
				'If at first you don’t succeed...<> call it version 1.0',
				'Why did Microsoft PowerPoint cross the road?<>To get to the other slide!',
				'What did the computer do at lunchtime?<>Had a byte!',
				'Why did the computer keep sneezing?<>It had a virus!',
				'What did one toilet say to the other?<>You look a bit flushed.',
				'Why did the picture go to jail?<>Because it was framed.',
				'What did one wall say to the other wall?<>I’ll meet you at the corner.',
				'What do you call a boy named Lee that no one talks to?<>Lonely',
				'Why do bicycles fall over?<>Because they are two-tired!',
				'Why was the broom late?<>It over swept!',
				'What part of the car is the laziest?<>The wheels, because they are always tired!',
				'What’s the difference between a TV and a newspaper?<>Ever tried swatting a fly with a TV?',
				'What did one elevator say to the other elevator?<>I think I’m coming down with something!',
				'Why was the belt arrested?<>Because it held up some pants!',
				'What makes the calendar seem so popular?<>Because it has a lot of dates!',
				'Why did Mickey Mouse take a trip into space?He wanted to find Pluto!',
				'Why do you go to bed every night?<>Because the bed won’t come to you!',
				'What has four wheels and flies?<>A garbage truck!',
				'Why did the robber take a bath before he stole from the bank?<>He wanted to make a clean get away!',
				'Just watched a documentary about beavers.<>It was the best damn program I’ve ever seen.',
				'Slept like a log last night<>woke up in the fireplace.',
				'Why did the scarecrow win an award?<>Because he was outstanding in his field.',
				'Why does a chicken coop only have two doors? <>Because if it had four doors it would be a chicken sedan.',
				'What’s the difference between an African elephant and an Indian elephant? <>About 5000 miles',
				'Why did the coffee file a police report? <>It got mugged.',
				'What did the grape do when he got stepped on? <>He let out a little wine.',
				'How many apples grow on a tree? <>All of them.',
				'What name do you give a person with a rubber toe? <>Roberto',
				'Did you hear about the kidnapping at school? <>It’s fine, he woke up.',
				'Why do scuba divers fall backwards into the water? <>Because if they fell forwards they’d still be in the boat.',
				'How does a penguin build it’s house? <>Igloos it together.',
				'What do you call a man with a rubber toe?<>Roberto',
				'Did you hear about the restaurant on the moon?<>Great food, no atmosphere.',
				'Why was the belt sent to jail?<>For holding up a pair of pants!',
				'Did you hear about the scientist who was lab partners with a pot of boiling water?<>He had a very esteemed colleague.',
				'What happens when a frogs car dies?<>He needs a jump. If that doesn’t work he has to get it toad.',
				'What did the flowers do when the bride walked down the aisle?<>They rose.',
				'Why did the man fall down the well?<>Because he couldn’t see that well.',
				'My boss told me to have a good day...<>...so I went home.',
				'How can you tell it’s a dogwood tree?<>By the bark.',
				'Did you hear about the kidnapping at school?<>It’s fine, he woke up.',
				'Why is Peter Pan always flying?<>Because he Neverlands.',
				'Which state has the most streets?<>Rhode Island.',
				'What do you call 26 letters that went for a swim?<>Alphawetical.',
				'Why was the color green notoriously single?<>It was always so jaded.',
				'Why did the coach go to the bank?<>To get his quarterback.',
				'How do celebrities stay cool?<>They have many fans.',
				'What’s the most depressing day of the week?<>sadder day.',
				'Dogs can’t operate MRI machines<>But catscan.',
				'I was going to tell a time-traveling joke<>but you guys didn’t like it.',
				'Stop looking for the perfect match<>instead look for a lighter.',
				'I told my doctor I heard buzzing<>but he said it’s just a bug going around.',
				'What kind of car does a sheep like to drive?<>A lamborghini.',
				'What did the accountant say while auditing a document?<>This is taxing.',
				'What did the two pieces of bread say on their wedding day?<>It was loaf at first sight.',
				'Why do melons have weddings?<>Because they cantaloupe.',
				'What did the drummer call his twin daughters?<>Anna One, Anna Two!',
				'What do you call a toothless bear?<> A gummy bear!',
				'Two goldfish are in a tank. <>One says to the other, “Do you know how to drive this thing?”',
				'What’s Forrest Gump’s password?<>1forrest1',
				'What is a child guilty of if they refuse to nap?<> Resisting a rest.',
				'I know a lot of jokes about retired people<>but none of them work.',
				'Why are spiders so smart?<>They can find everything on the web.',
				'What has one head, one foot, and four legs?<> A bed.',
				'What does a house wear?<> Address.',
				'What’s red and smells like blue paint?<>Red paint.',
				'My son asked me to put his shoes on<> but I don’t think they’ll fit me.',
				'I’ve been bored recently, so I decided to take up fencing.<> The neighbors keep demanding that I put it back.',
				'What do you call an unpredictable camera?<>A loose Canon.',
				'Which U.S. state is known for its especially small soft drinks?<>Minnesota.',
				'What do sprinters eat before a race?<> Nothing—they fast.',
				'I’m so good at sleeping...<>I can do it with my eyes closed.',
				'People are usually shocked that I have a Police record.<>But I love their greatest hits!',
				'I told my girlfriend she drew on her eyebrows too high.<> She seemed surprised.',
				'What do you call a fibbing cat?<> A lion.',
				'Why shouldn’t you write with a broken pencil?<> Because it’s pointless.',
				'I like telling Dad jokes…<>sometimes he laughs.',
				'How do you weigh a millennial?<> In Instagrams.',
				'The wedding was so beautiful<>even the cake was in tiers.',
				'What’s the most patriotic sport?<> Flag football.',
			];

			// answer
			var unknownCommand = false;
			if (TechwaveUserMessageCount === 1) {
				botText = welcomeText;
			} else {
				if (TechwaveUserQuestion.slice(0, 1) === '/') {
					var question = TechwaveUserQuestion.substring(1);
					var regex = /pass \d/i;
					var countpass = 15;
					if (regex.test(question)) {
						countpass = question.split(' ')[1];
						question = 'pass';
					}
					regex = /joke \d/i;
					var countjoke = 1;
					if (regex.test(question)) {
						countjoke = question.split(' ')[1];
						question = 'joke';
					}
					if (allCommands.hasOwnProperty(question)) {
						var allOptions = allCommands[question];
						var type = allOptions.type;
						if (type === 'text') {
							botText = allOptions.text;
						} else if (type === 'url') {
							window.location.href = allOptions.url;
						} else if (type === 'password') {
							var possible = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789![]{}()%&*$#^<>~@|';
							var password = '';
							if (countpass > 1000) {
								botText += '<p>I don\'t think that you want to get this password. Maximum password characters are: 1000.</p><p>Your password with 1000 characters:</p>';
								countpass = 1000;
							}
							for (var i = 0; i < countpass; i++) {
								password += possible.charAt(Math.floor(Math.random() * possible.length));
							}
							password = FrenifyTechWave.escapeHTML(password);
							botText += '<frenify_uselect>' + password + '</frenify_uselect>';
						} else if (type === 'time') {
							var dt = new Date();
							var hh = dt.getHours() < 10 ? '0' + dt.getHours() : dt.getHours();
							var mm = dt.getMinutes() < 10 ? '0' + dt.getMinutes() : dt.getMinutes();
							var ss = dt.getSeconds() < 10 ? '0' + dt.getSeconds() : dt.getSeconds();
							botText = hh + ":" + mm + ":" + ss;
						} else if (type === 'clear') {
							$('.fn__chatbot .chat__item.active').html('');
						} else if (type === 'joke') {
							if (countjoke > 1) {
								var shuffled = FrenifyTechWave.shuffleArray(allJokes);
								var selected = shuffled.slice(0, countjoke);
								botText = '<ul>';
								if (countjoke >= 1 && countjoke <= allJokes.length) {
									for (var j = 0; j < countjoke; j++) {
										botText += '<li>' + selected[j] + '</li>';
									}
								}
								botText += '</ul>';
							} else {
								botText = allJokes[Math.floor(Math.random() * allJokes.length)];
							}


						} else if (type === 'commands') {
							botText = commands;
						}
						append = allOptions.append;
					} else {
						unknownCommand = true;
					}
				} else {
					unknownCommand = true;
				}
			}
			if (unknownCommand) {
				botText = '<p>I only understand some commands. Of course, this is a fixable problem. Buy this template and implement AI and that\'s it. Go to the template site where you can buy? Visit item\'s website: <a href="https://themeforest.net/user/frenify/portfolio" target="_blank">TechWave</a></p><p>Write <frenify_main>/commands</frenify_main> to list all available commands.';
			}


			// answer to question
			if (append) {
				$('.fn__chat_comment button').addClass('disabled');
				setTimeout(function () {
					$('.fn__chatbot .chat__item.active').append('<div class="chat__box bot__chat"><div class="author"><span>Frenify Bot</span></div><div class="chat"><frenify_typing><h3><span>Thinking...</frenify></h3></div></div>');
					if ($('.techwave_fn_intro').length) {
						$("html, body").animate({ scrollTop: $('#fn__chat_textarea').offset().top - $(window).height() + 100 });
					} else {
						$("html, body").animate({ scrollTop: $(document).height() - $(window).height() });
					}
				}, 100);
				setTimeout(function () {
					$('.fn__chatbot .chat__item.active .chat__box.bot__chat:last-child .chat').html(botText);
					$('.fn__chat_comment button').removeClass('disabled');
					if ($('.techwave_fn_intro').length) {
						$("html, body").animate({ scrollTop: $('#fn__chat_textarea').offset().top - $(window).height() + 100 });
					} else {
						$("html, body").animate({ scrollTop: $(document).height() - $(window).height() });
					}
				}, 2000);
			}
		},

		shuffleArray: function (array) {
			var currentIndex = array.length, randomIndex;
			while (currentIndex !== 0) {
				randomIndex = Math.floor(Math.random() * currentIndex);
				currentIndex--;
				[array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
			}

			return array;
		},

		escapeHTML: function (string) {
			var entityMap = {
				'&': '&amp;',
				'<': '&lt;',
				'>': '&gt;',
				'"': '&quot;',
				"'": '&#39;',
				'/': '&#x2F;',
				'`': '&#x60;',
				'=': '&#x3D;'
			};
			return String(string).replace(/[&<>"'`=\/]/g, function (s) {
				return entityMap[s];
			});
		},

		aiChatBotOptions: function () {
			$('.fn__chat_link').off().on('click', function () {
				var element = $(this);
				if (!element.hasClass('active')) {
					$('.fn__chat_link.active').removeClass('active');
					$('.fn__chatbot .chat__item.active').removeClass('active');
					element.addClass('active');
					$(element.attr('href')).addClass('active');
					TechwaveUserMessageCount = $(element.attr('href')).find('.chat__box').length;
					$('.fn__new_chat_link').removeClass('active');
					$('.fn__chat_comment').removeClass('neww');
					$('.fn__chatbot .fn__title_holder .title').text(element.find('.text').text());
					if ($(element.attr('href')).html() === '') {
						$('.fn__chat_comment').addClass('neww');
					}
				}
				$('#fn__chat_textarea').frenifyMoveCursorToEnd();
				return false;
			});
			$('.fn__new_chat_link').off().on('click', function () {
				var element = $(this);
				if (!element.hasClass('active')) {
					$('.fn__chat_link.active').removeClass('active');
					$('.fn__chatbot .chat__item.active').removeClass('active');
					element.addClass('active');
					$(element.attr('href')).addClass('active');
					TechwaveUserMessageCount = 0;
					$('.fn__chatbot .fn__title_holder .title').text('New Chat');
				}
				$('.fn__chat_comment').addClass('neww');
				$('#fn__chat_textarea').frenifyMoveCursorToEnd();
				return false;
			});
			$('.fn__chat_link input').off().on('click', function (e) {
				e.stopPropagation();
			});
			$('.fn__chat_link .trigger').off().on('click', function () {
				var element = $(this),
					parent = element.closest('.fn__chat_link');
				if (parent.hasClass('opened')) {
					parent.removeClass('opened');
				} else {
					parent.addClass('opened');
				}
				return false;
			});

			$('.fn__chat_link .edit').off().on('click', function () {
				var element = $(this),
					parent = element.closest('.fn__chat_link'),
					input = parent.find('input');
				parent.addClass('live_edit').removeClass('opened');
				TechwaveInputText = input.val();
				setTimeout(function () {
					input.frenifyMoveCursorToEnd();
				}, 100);
				return false;
			});

			$('.fn__chat_link .cancel').off().on('click', function () {
				var e = $(this),
					parent = e.closest('.fn__chat_link'),
					input = parent.find('input');
				parent.removeClass('live_edit');
				input.val(TechwaveInputText);
				return false;
			});

			$('.fn__chat_link .save').off().on('click', function () {
				var e = $(this),
					parent = e.closest('.fn__chat_link'),
					input = parent.find('input');
				// do your ajax here
				parent.removeClass('live_edit');
				TechwaveInputText = input.val();
				parent.find('.text').text(TechwaveInputText);
				return false;
			});

			$(window).on('click', function () {
				$('.fn__chat_link').removeClass('opened');
			});

			$('.fn__chat_link .options__popup').on('click', function (e) {
				e.stopPropagation();
			});
		},

		aiChatBotTextareaHeight: function () {
			$('#fn__chat_textarea').on('mouseup keyup', function () {
				var e = $(this);
				var val = e.val();
				var padding = 34; // top 18 and bottom 16
				var border = 4; // top 2 and bottom 2
				var taLineHeight = 22; // This should match the line-height in the CSS
				var e2 = e.siblings('.fn__hidden_textarea');
				e2.val(val);
				var taHeight2 = e2[0].scrollHeight - padding; // Get the scroll height of the textarea
				var numberOfLines2 = Math.floor(taHeight2 / taLineHeight);
				e.css({ height: numberOfLines2 * taLineHeight + padding + border });
				if (numberOfLines2 > 6) {
					e.css({ overflowY: 'auto' });
				} else {
					e.css({ overflowY: 'hidden' });
				}
			});
			$('#fn__include_textarea').on('mouseup keyup', function () {
				var e = $(this);
				var val = e.val();
				var padding = 34; // top 18 and bottom 16
				var border = 4; // top 2 and bottom 2
				var taLineHeight = 22; // This should match the line-height in the CSS
				var e2 = e.siblings('.fn__hidden_textarea');
				e2.val(val);
				var taHeight2 = e2[0].scrollHeight - padding; // Get the scroll height of the textarea
				var numberOfLines2 = Math.floor(taHeight2 / taLineHeight);
				e.css({ height: numberOfLines2 * taLineHeight + padding + border });
				if (numberOfLines2 > 6) {
					e.css({ overflowY: 'auto' });
				} else {
					e.css({ overflowY: 'hidden' });
				}
			});
			$('#fn__exclude_textarea').on('mouseup keyup', function () {
				var e = $(this);
				var val = e.val();
				var padding = 34; // top 18 and bottom 16
				var border = 4; // top 2 and bottom 2
				var taLineHeight = 22; // This should match the line-height in the CSS
				var e2 = e.siblings('.fn__hidden_textarea');
				e2.val(val);
				var taHeight2 = e2[0].scrollHeight - padding; // Get the scroll height of the textarea
				var numberOfLines2 = Math.floor(taHeight2 / taLineHeight);
				e.css({ height: numberOfLines2 * taLineHeight + padding + border });
				if (numberOfLines2 > 6) {
					e.css({ overflowY: 'auto' });
				} else {
					e.css({ overflowY: 'hidden' });
				}
			});
		},

		billingProgress: function () {
			$('.techwave_fn_user_billing .progress').each(function () {
				var element = $(this);
				element.waypoint({
					handler: function () {
						if (!element.hasClass('active')) {
							setTimeout(function () {
								element.css('--frenify-progress', element.data('percentage'));
								element.addClass('active');
							}, 500);

						}
					},
					offset: '90%'
				});
			});
		},

		inputFileOnChange: function () {
			$('.fn__upload').on("change", function (event) {
				var element = $(this);
				var files = event.target.files;

				if (files.length > 0) {
					var length = files.length;
					var uploader = element.find(".upload_content");
					uploader.empty();
					var metasObj = element.closest(".acc__item").find(".metas");

					if (files.length == 1) {
						metasObj.show(true);
					}
					else {
						metasObj.hide(true);
					}

					for (var i = 0; i < files.length; i++) {
						uploader.append(`<p>${i + 1}. ${files[i].name}</p>`);
					};

					// uploader.append(`<p>${length} file(s) is selected.</p>`);
					element.addClass("has_file");

				}
			});
			$('.fn__upload .fn__closer').on('click', function () {
				var parent = $(this).closest('.fn__upload');
				parent.removeClass('has_img');
				parent.find('.preview_img').attr('src', '#');
				parent.find('input[type="file]').val('');
				return false;
			});
		},

		optionsList: function () {
			$('.fn__options_list a').off().on('click', function () {
				var e = $(this);
				if (e.hasClass('enabled')) {
					e.removeClass('enabled').addClass('disabled');
				} else {
					e.removeClass('disabled').addClass('enabled');
				}

				// do your ajax here
				// ....
				// ....

				return false;
			});
		},

		pricingTab: function () {
			$('.techwave_fn_pricing .toggle_in').each(function () {
				var element = $(this),
					active = element.find('.active');
				var offset = active.offset().left - element.offset().left;
				element.find('.bg').css({ left: offset, width: active.outerWidth(true, true) });
			});
			$('.techwave_fn_pricing .toggle_in a').off().on('click', function () {
				var element = $(this);
				if (!element.hasClass('active')) {
					var parent = element.closest('.toggle_in');
					var pricing = element.closest('.techwave_fn_pricing');
					var offset = element.offset().left - parent.offset().left;
					pricing.find('.pricing__tab.active').removeClass('active');
					$(element.attr('href')).addClass('active');
					element.siblings().removeClass('active');
					element.addClass('active');
					parent.find('.bg').css({ left: offset, width: element.outerWidth(true, true) });
				}
				return false;
			});
		},

		feedFilters: function () {

			// Enable/disable selecting items
			$('.techwave_fn_feed .filter__select input[type="checkbox"]').change(function () {
				var element = $(this);
				var checked = element.is(':checked');
				var feed = element.closest('.techwave_fn_feed');
				var items = feed.find('.fn__gallery_items .item');
				if (checked) {
					items.addClass('select__ready');
					feed.find('.fn__selection_box').slideDown(200);
				} else {
					items.removeClass('select__ready');
					feed.find('.fn__selection_box').slideUp(200);
				}
			});

			// select/deselect items
			$('.fn__selectable_item').off().on('click', function () {
				var element = $(this),
					page = element.closest('.techwave_fn_community_page'),
					items = page.find('.fn__gallery_items .item');
				if (element.hasClass('selected')) {
					element.removeClass('selected');
					TechwaveSelectedCount--;
				} else {
					element.addClass('selected');
					TechwaveSelectedCount++;
				}
				page.find('.fn__selection_box .count').text(TechwaveSelectedCount);
				return false;
			});


			// tab filter
			$('.techwave_fn_feed .fn__tabs a').on('click', function () {
				var element = $(this);
				if (!element.hasClass('active') && !TechwaveFeedFilterLoading) {
					TechwaveFeedFilterLoading = true;
					element.siblings().removeClass('active');
					element.addClass('active');
					var feed = element.closest('.techwave_fn_feed');
					feed.find('.feed__results').addClass('loading');

					// do your ajax here
					// you have to get new feeds with filter (if you want to change also filter) of selected tab via ajax
					// ....
					// ....
					// after ajax ends remove setTimeout (it was added just for HTML)

					setTimeout(function () {
						// here you have to insert your feeds into the filter's content and feed content
						feed.find('.feed__results').removeClass('loading');
						TechwaveFeedFilterLoading = false;
						FrenifyTechWave.galleryIsotope();
					}, 1500);

				}
				return false;
			});

			// Trending&New Filter
			$('.techwave_fn_feed .filter__sorting a').on('click', function () {
				var element = $(this);
				if (!element.hasClass('enabled') && !TechwaveFeedFilterLoading) {
					TechwaveFeedFilterLoading = true;
					element.siblings().removeClass('enabled').addClass('disabled');
					element.removeClass('disabled').addClass('enabled');
					var feed = element.closest('.techwave_fn_feed');
					feed.find('.feed__results').addClass('loading');

					// do your ajax here
					// you have to get new feeds by trending or new (selected) via ajax
					// ....
					// ....
					// after ajax ends remove setTimeout (it was added just for HTML)

					setTimeout(function () {
						// here you have to insert new feeds into the results' content
						feed.find('.feed__results').removeClass('loading');
						TechwaveFeedFilterLoading = false;
					}, 1500);
				}


				return false;
			});


			// Upscaled Filter
			$('.techwave_fn_feed .filter__upscaled input[type="checkbox"]').change(function () {
				var element = $(this);
				var checked = element.is(':checked');
				var feed = element.closest('.techwave_fn_feed');
				feed.find('.feed__results').addClass('loading');


				// do your ajax here
				// you have to get new feeds by checked value via ajax
				// ....
				// ....
				// after ajax ends remove setTimeout (it was added just for HTML)


				setTimeout(function () {
					// here you have to insert new feeds into the results' content
					feed.find('.feed__results').removeClass('loading');
				}, 1500);
			});

			// search filter
			$('.techwave_fn_feed .filter__search a').on('click', function () {
				if (!TechwaveFeedFilterLoading) {
					var feed = $(this).closest('.techwave_fn_feed');
					feed.find('.feed__results').addClass('loading');

					// do your ajax here
					// you have to get new feeds by search word via ajax
					// ....
					// ....
					// after ajax ends remove setTimeout (it was added just for HTML)

					setTimeout(function () {
						// here you have to insert new feeds into the results' content
						feed.find('.feed__results').removeClass('loading');
						TechwaveFeedFilterLoading = false;
					}, 1500);
				}

				return false;
			});
		},

		report: function () {
			var reportbox = $('.techwave_fn_report');
			$('.fn__report').off().on('click', function () {
				var e = $(this),
					id = e.data('id');

				if (reportbox.hasClass('opened')) {
					reportbox.removeClass('opened');
				} else {
					reportbox.addClass('opened');
				}
				return false;
			});


			reportbox.find('.cancel').off().on('click', function () {
				reportbox.removeClass('opened');
				return false;
			});
			reportbox.find('.fn__closer').off().on('click', function () {
				reportbox.removeClass('opened');
				return false;
			});
			reportbox.find('.report__closer').off().on('click', function () {
				reportbox.removeClass('opened');
				return false;
			});
		},

		follow: function () {
			$('.fn__follow').off().on('click', function () {
				var e = $(this),
					text = e.find('.text'),
					id = e.data('id');
				if (e.hasClass('has__follow')) {
					e.removeClass('has__follow');
					text.text(e.data('follow-text'));
				} else {
					e.addClass('has__follow');
					text.text(e.data('unfollow-text'));
				}
				return false;
			});
		},

		copyLink: function () {
			$(".fn__copy").off().on("click", function () {
				var e = $(this);
				var text = e.text();
				var copied = e.data("copied");
				var copy2 = e.attr("data-text");
				var copy = e.attr("href");
				if (typeof copy2 !== 'undefined' && copy2 !== false) {
					copy = copy2;
				}
				var temp = $("<input>");
				$("body").append(temp);
				temp.val(copy).select();
				document.execCommand("copy");
				temp.remove();
				e.text(copied).delay(1000).queue(function (nxt) {
					e.text(text);
					nxt();
				});
				return false;
			});
		},

		galleryIsotope: function () {

			var masonry = $('.fn__gallery_items');
			if ($().isotope) {
				masonry.each(function () {
					$(this).isotope({
						percentPosition: true,
						itemSelector: '.fn__gallery_item',
						masonry: {}
					});
				});
			}
		},

		imageLightbox: function () {
			var body = $('body');
			var scrollY = 0;
			$('.fn__gallery_items .item').off().on('click', function () {
				var element = $(this),
					id = element.data('id');
				if (!element.hasClass('select__ready')) {
					lightbox.scrollTop(0);
					// with this id you can create ajax to call this image into the lightbox
					scrollY = document.documentElement.style.getPropertyValue('--techwave-scroll-y');
					body.css({ position: 'fixed', top: scrollY });

					body.addClass('fn__lightbox_mode');
					lightbox.addClass('opened');
				}



				return false;
			});
			var lightbox = $('.techwave_fn_img_lightbox');

			lightbox.find('.fn__closer').off().on('click', function () {
				body.removeClass('fn__lightbox_mode');
				lightbox.removeClass('opened');
				body.css({ position: 'relative', top: '' });
				setTimeout(function () {
					window.scrollTo({ top: 300, left: 0, behavior: "instant" });
					FrenifyTechWave.galleryIsotope();
				}, 1);
			});
		},

		bookmark: function () {
			$('.fn__bookmark').off().on('click', function () {
				var e = $(this);
				if (e.hasClass('has__bookmark')) {
					e.removeClass('has__bookmark');
				} else {
					e.addClass('has__bookmark');
				}
				// do your ajax here
				return false;
			});
		},

		like: function () {
			$('.fn__like').off().on('click', function () {
				var e = $(this),
					countbox = e.find('.count'),
					id = e.data('id');
				if (e.hasClass('has__like')) {
					e.removeClass('has__like');
					countbox.text(parseInt(countbox.text()) - 1);
				} else {
					e.addClass('has__like');
					countbox.text(parseInt(countbox.text()) + 1);
				}
				// do your ajax here
				return false;
			});
		},

		accordion: function () {
			$('.techwave_fn_accordion').each(function () {
				$(this).find('.opened .acc__content').slideDown(300);
			});
			$('.techwave_fn_accordion .acc__header').on('click', function () {
				var element = $(this),
					parent = element.closest('.acc__item'),
					accordion = element.closest('.techwave_fn_accordion'),
					content = parent.find('.acc__content'),
					type = accordion.data('type');
				if (parent.hasClass('opened')) {
					parent.removeClass('opened');
					content.slideUp(300);
				} else {
					if (type === 'accordion') {
						accordion.find('.acc__item').removeClass('opened');
						accordion.find('.acc__content').slideUp(300);
					}
					parent.addClass('opened');
					content.slideDown(300);
				}
			});
		},

		search: function () {
			var searchBar = $('.techwave_fn_searchbar');
			var input = searchBar.find('.search__input');
			var resultsBar = searchBar.find('.search__results');
			var searchOpener = $('.fn__nav_bar .bar__item_search .item_opener');

			// open searchbar
			searchOpener.on('click', function () {
				searchBar.addClass('opened');
				setTimeout(function () {
					input[0].focus();
				}, 100);

				return false;
			});

			// close searchbar
			searchBar.find('.search__closer').on('click', function () {
				input.val('');
				resultsBar.removeClass('opened');
				searchBar.removeClass('opened');
				return false;
			});


			// search something
			var timeout = null;
			input.on('keyup', function () {
				var field = $(this);
				var text = field.val();

				clearTimeout(timeout);

				timeout = setTimeout(function () {
					if (text === '') {
						resultsBar.removeClass('opened');
					} else {
						resultsBar.addClass('opened');
						// add your ajax code here
					}
				}, 700);
			});
		},

		animatedText: function () {
			$('.fn__animated_text').each(function () {
				var element = $(this);
				var text = element.text();
				var letters = text.split('');
				var time = element.data('wait');
				if (!time) { time = 0; }
				var speed = element.data('speed');
				if (!speed) { speed = 4; }
				speed = speed / 100;
				element.html('<em>321...</em>').addClass('ready');

				element.waypoint({
					handler: function () {
						if (!element.hasClass('stop')) {
							element.addClass('stop');
							setTimeout(function () {
								element.text('');
								$.each(letters, function (e, i) {
									var span = document.createElement("span");
									span.textContent = i;
									span.style.animationDelay = e * speed + 's';
									element.append(span);
								});
							}, time);
						}
					},
					offset: '90%'
				});

			});
		},

		movingSubMenuForLeftPanel: function () {
			var fixedsub = $('.techwave_fn_fixedsub');
			var li = $('.techwave_fn_leftpanel .group__list > li');
			var rightpart = $('.techwave_fn_content');



			li.on('mouseenter', function () {
				var parentLi = $(this);
				var subMenu = parentLi.children('ul.sub-menu');
				var subMenuHtml = subMenu.html();
				//parentLi;
				if (subMenu.length) {
					li.removeClass('hovered');
					parentLi.addClass('hovered').parent().addClass('hovered');
					fixedsub.removeClass('opened').children('ul').html('').html(subMenuHtml);
					fixedsub.addClass('opened');
				} else {
					li.removeClass('hovered');
					fixedsub.removeClass('opened');
					parentLi.removeClass('hovered').parent().removeClass('hovered');
				}
				var topOffSet = parentLi.offset().top;
				var menuBar = $('.techwave_fn_leftpanel .leftpanel_content');
				var menuBarOffSet = menuBar.offset().top;
				var asd = topOffSet - menuBarOffSet;

				fixedsub.css({ top: asd });
				abc();
			});
			function abc() {
				rightpart.on('mouseenter', function () {
					fixedsub.removeClass('opened');
					li.removeClass('hovered').parent().removeClass('hovered');
				});
			}
			abc();
		},

		panelResize: function () {
			var wrapper = $('html');
			$('.techwave_fn_leftpanel .desktop_closer').off().on('click', function () {
				if (wrapper.hasClass('panel-opened')) {
					wrapper.removeClass('panel-opened');
					localStorage.frenify_panel = '';
				} else {
					wrapper.addClass('panel-opened');
					localStorage.frenify_panel = 'panel-opened';
				}
				return false;
			});
			$('.techwave_fn_leftpanel .mobile_closer').off().on('click', function () {
				if (wrapper.hasClass('mobile-panel-opened')) {
					wrapper.removeClass('mobile-panel-opened');
				} else {
					wrapper.addClass('mobile-panel-opened');
				}
				return false;
			});
		},

		navBarItems: function () {

			// user details
			var userItem = $('.fn__nav_bar .bar__item_user');
			userItem.find('.user_opener').on('click', function (e) {
				e.stopPropagation();
				if (userItem.hasClass('opened')) {
					userItem.removeClass('opened');
				} else {
					userItem.addClass('opened');
				}
				// close lightboxes
				$('.bar__item_language,.bar__item_notification').removeClass('opened');
				return false;
			});

			userItem.on('click', function (e) {
				e.stopPropagation();
			});

			$(window).on('click', function () {
				userItem.removeClass('opened');
			});

			// light and dark mode
			var darklightSwitcher = $('.fn__nav_bar .bar__item_skin .item_opener');
			darklightSwitcher.off().on('click', function () {
				if ($('html').attr('data-techwave-skin') === 'light') {
					$('html').attr('data-techwave-skin', 'dark');
					localStorage.frenify_skin = 'dark';
				} else {
					$('html').attr('data-techwave-skin', 'light');
					localStorage.frenify_skin = 'light';
				}

				// close lightboxes
				$('.bar__item_user,.bar__item_language,.bar__item_notification').removeClass('opened');
				return false;
			});

			// language
			var languageItem = $('.fn__nav_bar .bar__item_language');
			languageItem.find('.item_opener').on('click', function (e) {
				e.stopPropagation();
				if (languageItem.hasClass('opened')) {
					languageItem.removeClass('opened');
				} else {
					languageItem.addClass('opened');
				}
				// close lightboxes
				$('.bar__item_user,.bar__item_notification').removeClass('opened');
				return false;
			});

			languageItem.on('click', function (e) {
				e.stopPropagation();
			});

			$(window).on('click', function () {
				languageItem.removeClass('opened');
			});

			// notifications
			var notificationItem = $('.fn__nav_bar .bar__item_notification');
			notificationItem.find('.item_opener').on('click', function (e) {
				e.stopPropagation();
				if (notificationItem.hasClass('opened')) {
					notificationItem.removeClass('opened');
				} else {
					notificationItem.addClass('opened');
				}
				// close lightboxes
				$('.bar__item_user,.bar__item_language').removeClass('opened');
				return false;
			});

			notificationItem.on('click', function (e) {
				e.stopPropagation();
			});

			$(window).on('click', function () {
				notificationItem.removeClass('opened');
			});
		},

		redetectFullScreen: function () {
			var fbtn = $('.fn__nav_bar .bar__item_fullscreen a');
			if (window.innerHeight === screen.height) {
				fbtn.addClass('full_screen');
			} else {
				fbtn.removeClass('full_screen');
			}
		},

		fullSCreen: function () {
			var fbtn = $('.fn__nav_bar .bar__item_fullscreen a');

			fbtn.off().on('click', function () {
				if (fbtn.hasClass('full_screen')) {
					fbtn.removeClass('full_screen');
					if (document.exitFullscreen) {
						document.exitFullscreen();
					} else if (document.msExitFullscreen) {
						document.msExitFullscreen();
					} else if (document.mozCancelFullScreen) {
						document.mozCancelFullScreen();
					} else if (document.webkitExitFullscreen) {
						document.webkitExitFullscreen();
					}
				} else {
					fbtn.addClass('full_screen');
					if (document.documentElement.requestFullscreen) {
						document.documentElement.requestFullscreen();
					} else if (document.documentElement.msRequestFullscreen) {
						document.documentElement.msRequestFullscreen();
					} else if (document.documentElement.mozRequestFullScreen) {
						document.documentElement.mozRequestFullScreen();
					} else if (document.documentElement.webkitRequestFullscreen) {
						document.documentElement.webkitRequestFullscreen(Element.ALLOW_KEYBOARD_INPUT);
					}
				}
				return false;
			});
		},

		navSubMenu: function () {
			$('.techwave_fn_leftpanel .menu-item-has-children > a').off().on('click', function () {
				var e = $(this),
					li = e.closest('li');
				if (li.hasClass('closed')) {
					li.removeClass('closed');
					li.children('ul').slideDown(200);
				} else {
					li.addClass('closed');
					li.children('ul').slideUp(200);
				}
				return false;
			});
		},

		// preloader: function () {
		// 	var preloader = $('.techwave_fn_preloader');

		// 	var date2 = new Date();
		// 	var difference = date2 - FrenifyTechWaveTime;
		// 	var waitTime = 4000;
		// 	if (difference < waitTime) {
		// 		waitTime -= difference;
		// 	} else {
		// 		waitTime = 0;
		// 	}
		// 	if (!preloader.hasClass('wait_for_full_preloading_animation')) {
		// 		waitTime = 0;
		// 	}
		// 	setTimeout(function () {
		// 		preloader.addClass('fn_ready');
		// 	}, waitTime);
		// 	setTimeout(function () {
		// 		preloader.remove();
		// 	}, waitTime + 2000);
		// },

		imgToSVG: function () {
			$('img.fn__svg').each(function () {
				var img = $(this);
				var imgClass = img.attr('class');
				var imgURL = img.attr('src');

				$.get(imgURL, function (data) {
					var svg = $(data).find('svg');
					if (typeof imgClass !== 'undefined') {
						svg = svg.attr('class', imgClass + ' replaced-svg');
					}
					img.replaceWith(svg);

				}, 'xml');

			});
		},

		BgImg: function () {
			var div = $('*[data-bg-img]');
			div.each(function () {
				var element = $(this);
				var attrBg = element.attr('data-bg-img');
				var dataBg = element.data('bg-img');
				if (typeof (attrBg) !== 'undefined') {
					element.css({ backgroundImage: 'url(' + dataBg + ')' });
				}
			});
		},

	};


	// READY Functions
	$(document).ready(function () {
		FrenifyTechWave.init();
		$(':root').css('--techwave-scroll-y', (window.scrollY * (-1)) + 'px');
		setTimeout(function () {
			FrenifyTechWave.galleryIsotope();
		}, 500);
	});

	// RESIZE Functions
	$(window).on('resize', function () {
		FrenifyTechWave.popupMobile();
		FrenifyTechWave.redetectFullScreen();
		FrenifyTechWave.galleryIsotope();
	});

	//	$(window).load( function(){
	//		FrenifyTechWave.preloader();
	//		FrenifyTechWave.galleryIsotope();
	//		setTimeout(function(){
	//			FrenifyTechWave.galleryIsotope();
	//		},1000);
	//	});

	// LOAD Functions
	$(window).on('load', function () {
		FrenifyTechWave.preloader();
		FrenifyTechWave.galleryIsotope();
		setTimeout(function () {
			FrenifyTechWave.galleryIsotope();
		}, 1000);
	});

	$(window).on('scroll', function () {
		$(':root').css('--techwave-scroll-y', (window.scrollY * (-1)) + 'px');
	});

})(jQuery);
